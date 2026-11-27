"""指标汇总。

读取 events.jsonl 和 scenario.jsonl，算出每个请求的排队、执行、通知、响应时间，
再按整体、长短请求、首步和后续步骤、故障类型分组统计。

顶层时间点：
  arrive        生成器准备发起请求
  enqueue       请求进入 LLM 队列
  batch         调度器把请求取出，准备成批执行
  exec_start    后端工作线程真正开始执行
  backend_done  假后端结束睡眠
  notify        结果写回，调用方被唤醒
  return        调用方拿到结果
"""

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

METRIC_NAMES = [
    "response_ms",
    "enqueue_gap_ms",
    "batch_wait_ms",
    "sched_wait_ms",
    "service_ms",
    "notify_lag_ms",
    "slowdown",
]


def _percentile(values: List[float], quantile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def _stats(values: Iterable[Optional[float]]) -> Dict[str, Any]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {"n": 0, "mean": None, "median": None, "p90": None, "max": None}
    return {
        "n": len(clean),
        "mean": round(sum(clean) / len(clean), 3),
        "median": round(_percentile(clean, 0.5), 3),
        "p90": round(_percentile(clean, 0.9), 3),
        "max": round(max(clean), 3),
    }


def _load_events(path: Path) -> Dict[str, Dict[str, Any]]:
    """按请求收集时间点和结果字段。"""
    events: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            request_id = record.get("bench_request_id")
            if not request_id:
                continue
            entry = events.setdefault(request_id, {"ts": {}, "result": {}})
            name = record.get("event")
            entry["ts"].setdefault(name, []).append(record["ts"])
            if name == "return":
                entry["result"] = {
                    "status_code": record.get("status_code"),
                    "error": record.get("error"),
                }
    return events


def _first(entry: Optional[Dict[str, Any]], name: str) -> Optional[float]:
    if not entry:
        return None
    values = entry["ts"].get(name)
    return values[0] if values else None


def _diff(end: Optional[float], start: Optional[float]) -> Optional[float]:
    if end is None or start is None:
        return None
    return (end - start) * 1000.0


def _peak_concurrency(entries: List[Dict[str, Any]]) -> int:
    points: List[Tuple[float, int]] = []
    for entry in entries:
        start = _first(entry, "exec_start")
        end = _first(entry, "backend_done")
        if start is not None and end is not None and end >= start:
            points.append((start, 1))
            points.append((end, -1))

    points.sort(key=lambda item: (item[0], -item[1]))
    current = 0
    peak = 0
    for _, delta in points:
        current += delta
        peak = max(peak, current)
    return peak


def _batch_sizes(path: Path) -> List[int]:
    sizes: List[int] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("event") == "batch_dispatch":
                sizes.append(int(record.get("batch_size", 0)))
    return sizes


def _request_metrics(run: str, rows: List[Dict[str, Any]], events: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for row in rows:
        request_id = row["request_id"]
        entry = events.get(request_id)
        result = (entry or {}).get("result", {})

        arrive = _first(entry, "arrive")
        enqueue = _first(entry, "enqueue")
        batch = _first(entry, "batch")
        exec_start = _first(entry, "exec_start")
        backend_done = _first(entry, "backend_done")
        notify = _first(entry, "notify")
        returned = _first(entry, "return")

        service = _diff(backend_done, exec_start)
        response = _diff(returned, arrive)
        slowdown = response / service if response is not None and service else None

        records.append(
            {
                "run": run,
                "request_id": request_id,
                "task_id": row["task_id"],
                "step_index": row["step_index"],
                "num_steps": row.get("num_steps"),
                "fault": row.get("fault"),
                "prompt_tokens": row.get("prompt_tokens"),
                "output_tokens": row.get("output_tokens"),
                "planned_service_ms": row.get("service_ms"),
                "enqueue_gap_ms": _diff(enqueue, arrive),
                "batch_wait_ms": _diff(batch, enqueue),
                "sched_wait_ms": _diff(exec_start, batch),
                "service_ms": service,
                "notify_lag_ms": _diff(notify, backend_done),
                "response_ms": response,
                "slowdown": slowdown,
                "delivered": returned is not None,
                "notify_count": len((entry or {}).get("ts", {}).get("notify", [])),
                "status_code": result.get("status_code"),
                "error": result.get("error"),
                "_arrive": arrive,
                "_return": returned,
            }
        )
    return records


def _task_metrics(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tasks: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for record in records:
        tasks.setdefault((record["run"], record["task_id"]), []).append(record)

    result: List[Dict[str, Any]] = []
    for (run, task_id), items in tasks.items():
        arrives = [item["_arrive"] for item in items if item["_arrive"] is not None]
        returns = [item["_return"] for item in items if item["_return"] is not None]
        task_response = None
        if arrives and returns:
            task_response = (max(returns) - min(arrives)) * 1000.0

        result.append(
            {
                "run": run,
                "task_id": task_id,
                "steps": len(items),
                "delivered": sum(1 for item in items if item["delivered"]),
                "task_response_ms": task_response,
            }
        )
    return result


def _summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {name: _stats(record.get(name) for record in records) for name in METRIC_NAMES}


def _group_summary(records: List[Dict[str, Any]], tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    services = [record["planned_service_ms"] for record in records if record.get("planned_service_ms") is not None]
    service_cutoff = _percentile(services, 0.5) if services else None

    def classify(record: Dict[str, Any]) -> str:
        if service_cutoff is None or record.get("planned_service_ms") is None:
            return "unknown"
        return "short" if record["planned_service_ms"] <= service_cutoff else "long"

    groups: Dict[str, List[Dict[str, Any]]] = {"all": records}
    for record in records:
        groups.setdefault(f"service:{classify(record)}", []).append(record)
        step = "first_step" if record["step_index"] == 0 else "later_step"
        groups.setdefault(f"step:{step}", []).append(record)
        groups.setdefault(f"fault:{record.get('fault') or 'ok'}", []).append(record)

    summary: Dict[str, Any] = {}
    for name, items in groups.items():
        summary[name] = {
            "count": len(items),
            "delivered": sum(1 for item in items if item["delivered"]),
            "notify_missing": sum(1 for item in items if item["notify_count"] == 0),
            "notify_duplicate": sum(1 for item in items if item["notify_count"] > 1),
            "metrics": _summarize(items),
        }

    task_responses = [task["task_response_ms"] for task in tasks if task["task_response_ms"] is not None]
    summary["tasks"] = {
        "count": len(tasks),
        "delivered_all_steps": sum(1 for task in tasks if task["delivered"] == task["steps"]),
        "task_response_ms": _stats(task_responses),
    }
    return summary


def _overview(records: List[Dict[str, Any]], entries: List[Dict[str, Any]], batches: List[int]) -> Dict[str, Any]:
    arrives = [record["_arrive"] for record in records if record["_arrive"] is not None]
    returns = [record["_return"] for record in records if record["_return"] is not None]
    delivered = sum(1 for record in records if record["delivered"])

    span_ms = None
    if arrives and returns:
        span_ms = (max(returns) - min(arrives)) * 1000.0

    throughput = None
    if span_ms and span_ms > 0:
        throughput = delivered / (span_ms / 1000.0)

    batch_stats = {
        "count": len(batches),
        "mean_size": round(sum(batches) / len(batches), 3) if batches else None,
        "max_size": max(batches) if batches else None,
    }

    return {
        "requests": len(records),
        "delivered": delivered,
        "notify_missing": sum(1 for record in records if record["notify_count"] == 0),
        "notify_duplicate": sum(1 for record in records if record["notify_count"] > 1),
        "total_completion_ms": round(span_ms, 3) if span_ms is not None else None,
        "throughput_rps": round(throughput, 3) if throughput is not None else None,
        "peak_concurrency": _peak_concurrency(entries),
        "batches": batch_stats,
    }


def _write_requests(path: Path, records: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "run",
        "request_id",
        "task_id",
        "step_index",
        "num_steps",
        "fault",
        "prompt_tokens",
        "output_tokens",
        "planned_service_ms",
        "enqueue_gap_ms",
        "batch_wait_ms",
        "sched_wait_ms",
        "service_ms",
        "notify_lag_ms",
        "response_ms",
        "slowdown",
        "delivered",
        "notify_count",
        "status_code",
        "error",
    ]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({name: record.get(name) for name in fieldnames})


def _write_summary_csv(path: Path, summary: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["run", "group", "metric", "n", "mean", "median", "p90", "max"])
        for run, run_summary in summary["runs"].items():
            for group, payload in run_summary["groups"].items():
                if group == "tasks":
                    continue
                for metric, stats in payload["metrics"].items():
                    writer.writerow(
                        [run, group, metric, stats["n"], stats["mean"], stats["median"], stats["p90"], stats["max"]]
                    )
        for group, payload in summary["combined"]["groups"].items():
            if group == "tasks":
                continue
            for metric, stats in payload["metrics"].items():
                writer.writerow(
                    ["combined", group, metric, stats["n"], stats["mean"], stats["median"], stats["p90"], stats["max"]]
                )


def _run_dirs(root: Path) -> List[Path]:
    if (root / "events.jsonl").exists():
        return [root]
    return sorted(path for path in root.iterdir() if path.is_dir() and (path / "events.jsonl").exists())


def analyze(run_root: str) -> Dict[str, Any]:
    """汇总一次实验目录，写出 requests.csv、summary.csv、summary.json。"""
    root = Path(run_root)
    scenario_path = root / "scenario.jsonl"
    if not scenario_path.exists():
        raise FileNotFoundError(f"缺少场景文件: {scenario_path}")

    rows = []
    with open(scenario_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    all_records: List[Dict[str, Any]] = []
    all_tasks: List[Dict[str, Any]] = []
    all_entries: List[Dict[str, Any]] = []
    all_batches: List[int] = []
    runs: Dict[str, Any] = {}

    for run_dir in _run_dirs(root):
        run_name = run_dir.name
        events = _load_events(run_dir / "events.jsonl")
        batches = _batch_sizes(run_dir / "events.jsonl")
        entries = list(events.values())

        records = _request_metrics(run_name, rows, events)
        tasks = _task_metrics(records)
        runs[run_name] = {
            "overview": _overview(records, entries, batches),
            "groups": _group_summary(records, tasks),
        }

        all_records.extend(records)
        all_tasks.extend(tasks)
        all_entries.extend(entries)
        all_batches.extend(batches)

    combined_overview = _overview(all_records, all_entries, all_batches)
    if runs:
        combined_overview["peak_concurrency"] = max(
            run["overview"]["peak_concurrency"] for run in runs.values()
        )

    summary = {
        "runs": runs,
        "combined": {
            "overview": combined_overview,
            "groups": _group_summary(all_records, all_tasks),
        },
    }

    _write_requests(root / "requests.csv", all_records)
    _write_summary_csv(root / "summary.csv", summary)
    with open(root / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    return summary
