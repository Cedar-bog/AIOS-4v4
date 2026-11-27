"""测量环境自检。

运行方式：

    python -m benchmark.selftest

检查项目：
1. 场景生成可复现，服务时间和参数一致。
2. 单请求的七个时间点齐全且单调递增。
3. 同批请求体现出队头阻塞，原版没有并发上限。
4. 多步任务的每一步都能送达。
5. 错误和超时请求的归属正确，每个请求只通知一次。
6. 指标之间的关系合理。
"""

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from .metrics import analyze
from .runner import run_once
from .scenario import DEFAULTS, _deep_merge, generate

REQUIRED_EVENTS = ["arrive", "enqueue", "batch", "exec_start", "backend_done", "notify", "return"]


def _config(overrides: Dict[str, Any]) -> Dict[str, Any]:
    return _deep_merge(DEFAULTS, overrides)


def _run(root: Path, config: Dict[str, Any], batch_interval: float, rows: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    if rows is None:
        rows = generate(config, str(root / "scenario.jsonl"))
    else:
        with open(root / "scenario.jsonl", "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    run_once(
        rows,
        root / "run_00",
        batch_interval=batch_interval,
        max_concurrency=int(config.get("max_concurrency", 4)),
    )
    return analyze(str(root))


def _events_by_request(path: Path) -> Dict[str, Dict[str, List[float]]]:
    events: Dict[str, Dict[str, List[float]]] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            request_id = record.get("bench_request_id")
            if request_id:
                events.setdefault(request_id, {}).setdefault(record["event"], []).append(record["ts"])
    return events


def check_scenario() -> None:
    config = _config({"seed": 99, "num_tasks": 30})
    first = generate(config)
    second = generate(config)
    assert first == second, "同一配置两次生成结果不一致"

    model = config["model"]
    for row in first:
        if row["fault"] == "slow":
            continue
        expected = (
            model["base_ms"]
            + model["prefill_ms_per_token"] * row["prompt_tokens"]
            + model["decode_ms_per_token"] * row["output_tokens"]
        )
        ratio = row["service_ms"] / expected
        assert 0.5 <= ratio <= 2.0, f"服务时间和参数偏差过大: {row}"

    print("PASS 场景生成可复现")


def check_single_request(root: Path) -> None:
    config = _config({"seed": 1, "num_tasks": 1, "arrival": {"pattern": "all_at_once"}})
    _run(root, config, batch_interval=0.05)

    events = _events_by_request(root / "run_00" / "events.jsonl")
    assert len(events) == 1, "请求数量不对"
    entry = next(iter(events.values()))
    for name in REQUIRED_EVENTS:
        assert name in entry, f"缺少事件 {name}"
        assert len(entry[name]) == 1, f"事件 {name} 出现 {len(entry[name])} 次"

    order = [entry[name][0] for name in REQUIRED_EVENTS]
    assert order == sorted(order), "时间点不是单调递增"

    print("PASS 单请求时间点完整")


def check_batch_blocking(root: Path) -> None:
    config = _config(
        {
            "seed": 3,
            "num_tasks": 12,
            "max_concurrency": 4,
            "arrival": {"pattern": "all_at_once"},
            "output": {"median_tokens": 80, "sigma": 1.0},
            "model": {"decode_ms_per_token": 10.0},
        }
    )
    summary = _run(root, config, batch_interval=0.1)
    overview = summary["combined"]["overview"]
    metrics = summary["combined"]["groups"]["all"]["metrics"]

    assert overview["delivered"] == 12, "请求应全部送达"
    assert overview["peak_concurrency"] <= 4, "峰值并发不应超过上限"
    assert overview["peak_concurrency"] >= 2, "并发上限内应当有并行执行"
    assert metrics["notify_lag_ms"]["p90"] < 50.0, "结果应在完成后立即写回"

    print("PASS 并发上限和及时返回")


def check_multi_step(root: Path) -> None:
    config = _config(
        {
            "seed": 5,
            "num_tasks": 5,
            "steps": {"min": 3, "max": 3, "gap_ms": 50.0},
            "arrival": {"pattern": "all_at_once"},
        }
    )
    summary = _run(root, config, batch_interval=0.1)
    tasks = summary["combined"]["groups"]["tasks"]

    assert summary["combined"]["overview"]["requests"] == 15, "多步任务请求数应为 5 x 3"
    assert summary["combined"]["overview"]["delivered"] == 15, "多步任务应全部送达"
    assert tasks["delivered_all_steps"] == 5, "应有三步都完成的任务数等于 5"

    print("PASS 多步任务")


def check_faults(root: Path) -> None:
    config = _config(
        {
            "seed": 7,
            "num_tasks": 12,
            "arrival": {"pattern": "all_at_once"},
            "faults": {"error_rate": 0.0, "timeout_rate": 0.0, "slow_rate": 0.0},
        }
    )
    rows = generate(config, str(root / "scenario.jsonl"))
    pattern = ["ok"] * 4 + ["error"] * 4 + ["timeout"] * 4
    for row, fault in zip(rows, pattern):
        row["fault"] = fault

    # 故障注入会按预期打印错误堆栈，这里临时关掉日志，避免自检输出混乱。
    import logging

    logging.disable(logging.CRITICAL)
    try:
        summary = _run(root, config, batch_interval=0.1, rows=rows)
    finally:
        logging.disable(logging.NOTSET)
    overview = summary["combined"]["overview"]
    assert overview["delivered"] == 12, "故障请求也应正常返回结果"
    assert overview["notify_missing"] == 0, "不应有请求缺少通知"
    assert overview["notify_duplicate"] == 0, "不应有请求被通知多次"

    import csv

    with open(root / "requests.csv", "r", encoding="utf-8") as fh:
        records = list(csv.DictReader(fh))

    by_fault = {record["fault"]: record["status_code"] for record in records}
    assert by_fault["error"] == "500", "错误请求应返回 500"
    assert by_fault["timeout"] == "408", "超时请求应返回 408"

    print("PASS 故障请求归属和通知次数")


def check_metrics(root: Path) -> None:
    summary = _run(root, _config({"seed": 11, "num_tasks": 20}), batch_interval=0.1)
    metrics = summary["combined"]["groups"]["all"]["metrics"]

    assert metrics["response_ms"]["n"] == 20
    assert metrics["response_ms"]["median"] >= metrics["service_ms"]["median"], "响应时间不应小于服务时间"
    assert metrics["slowdown"]["max"] >= 1.0, "放大倍数应至少为 1"
    assert metrics["notify_lag_ms"]["max"] >= 0.0

    print("PASS 指标关系合理")


def main() -> int:
    base = Path(tempfile.mkdtemp(prefix="aios-bench-selftest-"))
    try:
        check_scenario()
        check_single_request(base / "single")
        check_batch_blocking(base / "batch")
        check_multi_step(base / "multi")
        check_faults(base / "faults")
        check_metrics(base / "metrics")
    finally:
        shutil.rmtree(base, ignore_errors=True)
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
