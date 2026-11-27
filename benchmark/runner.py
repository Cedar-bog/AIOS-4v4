"""实验入口。

命令行示例：

    python -m benchmark.runner --config benchmark/configs/smoke.yaml --out results/smoke

一次运行包含：生成场景、安装探针、启动 FIFO 调度器、按场景发请求、停止调度器、
汇总指标。重复运行会写到 run_00、run_01 等子目录，共用同一份场景文件。
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from aios.llm_core.adapter import LLMAdapter
from aios.scheduler.bounded_scheduler import BoundedScheduler
from aios.scheduler.duration import ConstantPredictor, DurationPredictor, OraclePredictor
from aios.scheduler.policies import make_selection
from aios.syscall.syscall import create_syscall_executor
from aios.utils import probe

from .metrics import analyze
from .mock_backend import MockLLMAdapter
from .scenario import generate, load_config, load_scenario
from .tracer import JsonlSink
from .workload import run_scenario


class _NullManager:
    """占位管理器。

    实验只发 LLM 请求，内存、存储、工具三条路径不会被调度器调用，
    传空实现可以避免初始化真实后端。
    """

    def address_request(self, *args: Any, **kwargs: Any) -> None:
        return None


def _quiet_logging() -> None:
    """把日志降到 WARNING，避免同步 I/O 污染时间戳。"""
    logging.getLogger().setLevel(logging.WARNING)
    for name in ("aios", "litellm", "httpx", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)


def build_scheduler(
    batch_interval: float,
    llm: LLMAdapter,
    max_concurrency: int = 4,
    selection: Optional[Any] = None,
    predictor: Optional[Any] = None,
) -> BoundedScheduler:
    from aios.hooks.stores._global import (
        global_llm_req_queue_get_message,
        global_memory_req_queue_get_message,
        global_storage_req_queue_get_message,
        global_tool_req_queue_get_message,
    )

    return BoundedScheduler(
        llm=llm,
        memory_manager=_NullManager(),
        storage_manager=_NullManager(),
        tool_manager=_NullManager(),
        log_mode="silent",
        get_llm_syscall=global_llm_req_queue_get_message,
        get_memory_syscall=global_memory_req_queue_get_message,
        get_storage_syscall=global_storage_req_queue_get_message,
        get_tool_syscall=global_tool_req_queue_get_message,
        batch_interval=batch_interval,
        max_concurrency=max_concurrency,
        selection=selection,
        predictor=predictor,
    )


def run_once(
    rows: List[Dict[str, Any]],
    out_dir: Path,
    batch_interval: float,
    model_name: str = "mock-model",
    max_concurrency: int = 4,
    selection_policy: str = "fifo",
    predictor_mode: str = "sliding",
    cold_start_ms: float = 500.0,
    predictor_window: int = 32,
) -> Dict[str, Any]:
    """用给定场景跑一次，把事件写到 out_dir/events.jsonl。"""
    _quiet_logging()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sink = JsonlSink(str(out_dir / "events.jsonl"))
    probe.set_sink(sink)

    scheduler: Optional[BoundedScheduler] = None
    started_wall = time.time()
    error: Optional[str] = None
    summary: Dict[str, Any] = {}

    try:
        llm = MockLLMAdapter(model_name=model_name)
        if predictor_mode == "constant":
            predictor = ConstantPredictor(cold_start_ms)
        elif predictor_mode == "oracle":
            predictor = OraclePredictor(cold_start_ms)
        elif predictor_mode == "sliding":
            predictor = DurationPredictor(
                model_name=llm.llm_configs[0].name,
                window=predictor_window,
                cold_start_ms=cold_start_ms,
            )
        else:
            raise ValueError(f"未知的估计器模式: {predictor_mode}")
        selection = make_selection(selection_policy, predictor)
        scheduler = build_scheduler(
            batch_interval=batch_interval,
            llm=llm,
            max_concurrency=max_concurrency,
            selection=selection,
            predictor=predictor,
        )
        execute_request, _, _ = create_syscall_executor()

        scheduler.start()
        started_mono = time.monotonic()
        try:
            summary = run_scenario(rows, execute_request, start_mono=started_mono)
        except BaseException as exc:  # 记录后继续收尾，保证调度器被停掉
            error = repr(exc)
            raise
        finally:
            scheduler.stop()
            summary.setdefault("wall_ms", (time.monotonic() - started_mono) * 1000.0)
    finally:
        probe.clear_sink()
        sink.close()

    meta = {
        "events": str(out_dir / "events.jsonl"),
        "event_count": sink.count,
        "batch_interval": batch_interval,
        "max_concurrency": max_concurrency,
        "selection_policy": selection_policy,
        "predictor_mode": predictor_mode,
        "cold_start_ms": cold_start_ms,
        "predictor_window": predictor_window,
        "model_name": model_name,
        "started_wall": started_wall,
        "summary": {
            key: value
            for key, value in summary.items()
            if key != "failures"
        },
        "error": error,
    }
    with open(out_dir / "run_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    return meta


def run_experiment(
    config_path: Optional[str],
    out_root: str,
    batch_interval: Optional[float] = None,
    repeats: int = 1,
    model_name: str = "mock-model",
    max_concurrency: Optional[int] = None,
    selection_policy: Optional[str] = None,
    predictor_mode: Optional[str] = None,
    cold_start_ms: Optional[float] = None,
    predictor_window: Optional[int] = None,
    do_analyze: bool = True,
) -> Dict[str, Any]:
    """按配置跑整组实验，重复运行写到不同子目录。"""
    _quiet_logging()

    config = load_config(config_path)
    out_root_path = Path(out_root)
    out_root_path.mkdir(parents=True, exist_ok=True)

    if batch_interval is None:
        batch_interval = float(config.get("batch_interval", 0.1))

    if max_concurrency is None:
        max_concurrency = int(config.get("max_concurrency", 4))

    if selection_policy is None:
        selection_policy = str(config.get("selection_policy", "fifo"))

    if predictor_mode is None:
        predictor_mode = str(config.get("predictor_mode", "sliding"))

    if cold_start_ms is None:
        cold_start_ms = float(config.get("cold_start_ms", 500.0))

    if predictor_window is None:
        predictor_window = int(config.get("predictor_window", 32))

    scenario_path = out_root_path / "scenario.jsonl"
    rows = generate(config, str(scenario_path))
    # 后面的重复运行复用同一份场景，保证每个请求的服务时间一致。
    rows = load_scenario(str(scenario_path))

    overview = {
        "config_path": config_path,
        "config": config,
        "batch_interval": batch_interval,
        "max_concurrency": max_concurrency,
        "selection_policy": selection_policy,
        "predictor_mode": predictor_mode,
        "cold_start_ms": cold_start_ms,
        "predictor_window": predictor_window,
        "scenario": str(scenario_path),
        "requests": len(rows),
        "tasks": config.get("num_tasks"),
        "runs": [],
    }

    for index in range(repeats):
        run_dir = out_root_path / f"run_{index:02d}"
        meta = run_once(
            rows=rows,
            out_dir=run_dir,
            batch_interval=batch_interval,
            model_name=model_name,
            max_concurrency=max_concurrency,
            selection_policy=selection_policy,
            predictor_mode=predictor_mode,
            cold_start_ms=cold_start_ms,
            predictor_window=predictor_window,
        )
        overview["runs"].append(meta)
        print(
            f"[bench] run {index + 1}/{repeats} "
            f"events={meta['event_count']} "
            f"delivered={meta['summary'].get('delivered')}/"
            f"{meta['summary'].get('requests')} "
            f"wall={meta['summary'].get('wall_ms', 0) / 1000:.2f}s"
        )

    with open(out_root_path / "experiment.json", "w", encoding="utf-8") as fh:
        json.dump(overview, fh, ensure_ascii=False, indent=2)

    if do_analyze:
        analyze(str(out_root_path))

    return overview


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="运行 AIOS 调度对比实验")
    parser.add_argument("--config", required=True, help="场景配置 YAML 路径")
    parser.add_argument("--out", required=True, help="输出目录")
    parser.add_argument("--batch-interval", type=float, default=None, help="调度器批次间隔，秒")
    parser.add_argument("--max-concurrency", type=int, default=None, help="并发上限，默认读配置")
    parser.add_argument("--selection-policy", default=None, help="选择策略：fifo、shortest、hrrn")
    parser.add_argument("--predictor-mode", default=None, help="估计器：sliding、constant、oracle")
    parser.add_argument("--cold-start-ms", type=float, default=None, help="耗时估计冷启动值，毫秒")
    parser.add_argument("--predictor-window", type=int, default=None, help="滑动均值窗口")
    parser.add_argument("--repeats", type=int, default=1, help="重复次数")
    parser.add_argument("--model-name", default="mock-model", help="假后端模型名")
    parser.add_argument("--no-analyze", action="store_true", help="只跑实验，不做汇总")
    args = parser.parse_args(argv)

    run_experiment(
        config_path=args.config,
        out_root=args.out,
        batch_interval=args.batch_interval,
        repeats=args.repeats,
        model_name=args.model_name,
        max_concurrency=args.max_concurrency,
        selection_policy=args.selection_policy,
        predictor_mode=args.predictor_mode,
        cold_start_ms=args.cold_start_ms,
        predictor_window=args.predictor_window,
        do_analyze=not args.no_analyze,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
