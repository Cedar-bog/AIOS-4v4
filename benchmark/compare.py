"""两轮实验的按请求配对比较。

同一个场景在不同分支上生成的请求序列和服务时间完全一致，因此可以按
(run, request_id) 配对，计算每个请求的差值，再统计差值的中位数和 P90。
机器整体变慢时两边一起变慢，差值不受影响。

命令行：

    python -m benchmark.compare --base results/ab/baseline --opt results/ab/optimized
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

METRICS = ["response_ms", "notify_lag_ms", "batch_wait_ms", "sched_wait_ms", "service_ms"]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _num(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _percentile(values: List[float], quantile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def _stats(values: List[Optional[float]]) -> Dict[str, Any]:
    clean = [value for value in values if value is not None]
    if not clean:
        return {"n": 0, "median": None, "mean": None, "p90": None}
    return {
        "n": len(clean),
        "median": round(_percentile(clean, 0.5), 3),
        "mean": round(sum(clean) / len(clean), 3),
        "p90": round(_percentile(clean, 0.9), 3),
    }


def _paired_rows(base: List[Dict[str, str]], opt: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """按 (run, request_id) 配对，只保留两侧都有的请求。"""
    base_map = {(row["run"], row["request_id"]): row for row in base}
    opt_map = {(row["run"], row["request_id"]): row for row in opt}

    common = sorted(set(base_map) & set(opt_map))
    pairs: List[Dict[str, Any]] = []
    for key in common:
        base_row = base_map[key]
        opt_row = opt_map[key]
        item: Dict[str, Any] = {
            "run": key[0],
            "request_id": key[1],
            "step_index": int(base_row["step_index"]),
            "planned_service_ms": _num(base_row["planned_service_ms"]),
        }
        for metric in METRICS:
            base_value = _num(base_row.get(metric))
            opt_value = _num(opt_row.get(metric))
            item[f"base_{metric}"] = base_value
            item[f"opt_{metric}"] = opt_value
            item[f"delta_{metric}"] = (
                opt_value - base_value
                if base_value is not None and opt_value is not None
                else None
            )
        pairs.append(item)
    return pairs


def _groups(pairs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    services = [pair["planned_service_ms"] for pair in pairs if pair["planned_service_ms"] is not None]
    cutoff = _percentile(services, 0.5) if services else None

    groups: Dict[str, List[Dict[str, Any]]] = {"all": pairs}
    for pair in pairs:
        if cutoff is not None and pair["planned_service_ms"] is not None:
            name = "short" if pair["planned_service_ms"] <= cutoff else "long"
            groups.setdefault(f"service:{name}", []).append(pair)
        step = "first_step" if pair["step_index"] == 0 else "later_step"
        groups.setdefault(f"step:{step}", []).append(pair)
    return groups


def _summarize(pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for metric in METRICS:
        base_values = [pair[f"base_{metric}"] for pair in pairs]
        opt_values = [pair[f"opt_{metric}"] for pair in pairs]
        deltas = [pair[f"delta_{metric}"] for pair in pairs]
        result[metric] = {
            "base": _stats(base_values),
            "opt": _stats(opt_values),
            "delta": _stats(deltas),
        }
    return result


def compare(base_dir: str, opt_dir: str, out_dir: Optional[str] = None) -> Dict[str, Any]:
    base_root = Path(base_dir)
    opt_root = Path(opt_dir)

    base_rows = _read_csv(base_root / "requests.csv")
    opt_rows = _read_csv(opt_root / "requests.csv")

    pairs = _paired_rows(base_rows, opt_rows)
    if not pairs:
        raise RuntimeError("两个结果目录没有可配对的请求，检查场景是否一致")

    groups = _groups(pairs)
    summary: Dict[str, Any] = {
        "base": str(base_root),
        "opt": str(opt_root),
        "paired_requests": len(pairs),
        "groups": {name: _summarize(items) for name, items in groups.items()},
    }

    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "comparison.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
        with open(out / "comparison.csv", "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["group", "metric", "n", "base_median", "opt_median", "median_delta", "p90_delta", "mean_delta"]
            )
            for group, metrics in summary["groups"].items():
                for metric, payload in metrics.items():
                    writer.writerow(
                        [
                            group,
                            metric,
                            payload["delta"]["n"],
                            payload["base"]["median"],
                            payload["opt"]["median"],
                            payload["delta"]["median"],
                            payload["delta"]["p90"],
                            payload["delta"]["mean"],
                        ]
                    )

    return summary


def format_table(summary: Dict[str, Any], group: str = "all") -> str:
    metrics = summary["groups"].get(group, {})
    lines = [
        "指标                 基线中位数   优化中位数   差值中位数   差值P90",
        "--------------------  ----------  ----------  ----------  ----------",
    ]
    for metric in METRICS:
        payload = metrics.get(metric)
        if not payload:
            continue
        base = payload["base"]["median"]
        opt = payload["opt"]["median"]
        delta = payload["delta"]["median"]
        p90 = payload["delta"]["p90"]
        lines.append(
            f"{metric:<20}  {base if base is not None else '-':>10}  "
            f"{opt if opt is not None else '-':>10}  "
            f"{delta if delta is not None else '-':>10}  "
            f"{p90 if p90 is not None else '-':>10}"
        )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="按请求配对比较两轮实验")
    parser.add_argument("--base", required=True, help="基线结果目录，需包含 requests.csv")
    parser.add_argument("--opt", required=True, help="优化结果目录，需包含 requests.csv")
    parser.add_argument("--out", default=None, help="输出目录，写出 comparison.json 和 comparison.csv")
    args = parser.parse_args(argv)

    summary = compare(args.base, args.opt, args.out)
    print(format_table(summary, "all"))
    for group in ("service:short", "service:long"):
        if group in summary["groups"]:
            print()
            print(group)
            print(format_table(summary, group))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
