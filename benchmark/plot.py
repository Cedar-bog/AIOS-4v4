"""从实验结果生成图表。

用法：

    python -m benchmark.plot --results results/full --out results/figures

所有图保存为 PNG，不弹窗。只读取 comparison.json、summary.json，不重新计算指标。
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _load(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _comparisons(results: Path) -> List[tuple]:
    # bounded-multi-c8 用无上限版本作基线，比的是容量而不是补位，不纳入图表。
    excluded = {"bounded-multi-c8"}
    items = []
    for entry in sorted(results.iterdir()):
        if entry.name in excluded:
            continue
        comparison = entry / "comparison.json"
        if comparison.exists():
            items.append((entry.name, _load(comparison)))
    return items


def plot_overall(results: Path, out: Path) -> None:
    items = _comparisons(results)
    if not items:
        return

    names = [name for name, _ in items]
    base = [data["groups"]["all"]["response_ms"]["base"]["median"] for _, data in items]
    opt = [data["groups"]["all"]["response_ms"]["opt"]["median"] for _, data in items]

    positions = range(len(names))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar([p - 0.2 for p in positions], base, width=0.4, label="baseline")
    ax.bar([p + 0.2 for p in positions], opt, width=0.4, label="optimized")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("response median (ms)")
    ax.set_title("Response time median by experiment")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "response_median.png", dpi=150)
    plt.close(fig)

    delta = [
        data["groups"]["all"]["response_ms"]["delta"]["median"] for _, data in items
    ]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.barh(names, delta)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("median delta (ms), negative means faster")
    ax.set_title("Per-request paired change")
    fig.tight_layout()
    fig.savefig(out / "response_delta.png", dpi=150)
    plt.close(fig)


def plot_short_long(results: Path, out: Path, experiments: List[str]) -> None:
    items = {name: data for name, data in _comparisons(results)}
    selected = [name for name in experiments if name in items]
    if not selected:
        return

    fig, axes = plt.subplots(1, len(selected), figsize=(5 * len(selected), 4), sharey=False)
    if len(selected) == 1:
        axes = [axes]

    for ax, name in zip(axes, selected):
        data = items[name]["groups"]
        groups = ["service:short", "service:long"]
        labels = ["short", "long"]
        base = [data[g]["response_ms"]["base"]["median"] for g in groups]
        opt = [data[g]["response_ms"]["opt"]["median"] for g in groups]
        positions = range(len(groups))
        ax.bar([p - 0.2 for p in positions], base, width=0.4, label="baseline")
        ax.bar([p + 0.2 for p in positions], opt, width=0.4, label="optimized")
        ax.set_xticks(list(positions))
        ax.set_xticklabels(labels)
        ax.set_title(name)
        ax.set_ylabel("response median (ms)")

    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out / "short_long.png", dpi=150)
    plt.close(fig)


def plot_c_sweep(sweep_dir: Path, out: Path) -> None:
    if not sweep_dir.exists():
        return

    points = []
    for entry in sorted(sweep_dir.iterdir()):
        summary_path = entry / "summary.json"
        if not summary_path.exists() or not entry.name.startswith("c"):
            continue
        try:
            c = int(entry.name[1:])
        except ValueError:
            continue
        summary = _load(summary_path)
        runs = list(summary["runs"].values())
        throughput = sum(run["overview"]["throughput_rps"] for run in runs) / len(runs)
        median = summary["combined"]["groups"]["all"]["metrics"]["response_ms"]["median"]
        p90 = summary["combined"]["groups"]["all"]["metrics"]["response_ms"]["p90"]
        points.append((c, throughput, median, p90))

    if not points:
        return
    points.sort()
    cs = [p[0] for p in points]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(cs, [p[1] for p in points], marker="o")
    axes[0].set_xlabel("concurrency cap C")
    axes[0].set_ylabel("throughput (req/s)")
    axes[0].set_title("Throughput")

    axes[1].plot(cs, [p[2] for p in points], marker="o", label="median")
    axes[1].plot(cs, [p[3] for p in points], marker="s", label="P90")
    axes[1].set_xlabel("concurrency cap C")
    axes[1].set_ylabel("response time (ms)")
    axes[1].set_title("Response time")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out / "c_sweep.png", dpi=150)
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="生成实验结果图表")
    parser.add_argument("--results", default="results/full")
    parser.add_argument("--sweep", default="results/c-sweep")
    parser.add_argument("--out", default="results/figures")
    args = parser.parse_args(argv)

    results = Path(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    plot_overall(results, out)
    plot_short_long(
        results,
        out,
        [
            "refill-c8",
            "refill-c32",
            "refill-multi-c8",
            "hrrn-mixed",
            "hrrn-c20",
            "bounded-multi",
        ],
    )
    plot_c_sweep(Path(args.sweep), out)

    print(f"图表已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
