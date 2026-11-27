"""多轮 A/B 实验驱动。

在两个 worktree 上交替运行同一份场景，避免机器状态随时间漂移影响结论。
必须串行执行，不能并行，否则两边会抢 CPU。

示例：

    python -m benchmark.ab \
        --baseline-dir ../aios-baseline \
        --opt-dir ../aios-optimized \
        --config benchmark/configs/mixed.yaml \
        --rounds 3 \
        --out results/ab-mixed
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .compare import compare, format_table
from .metrics import analyze


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_rev(workdir: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(workdir),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _run_side(workdir: Path, config: str, out_dir: Path, batch_interval: float, model_name: str) -> None:
    cmd = [
        sys.executable,
        "-m",
        "benchmark.runner",
        "--config",
        config,
        "--out",
        str(out_dir),
        "--batch-interval",
        str(batch_interval),
        "--model-name",
        model_name,
        "--repeats",
        "1",
        "--no-analyze",
    ]
    subprocess.run(cmd, cwd=str(workdir), check=True)


def _collect(round_dirs: List[Path], dest: Path) -> None:
    """把多轮的 run_00/events.jsonl 合成一个可分析的目录。"""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    scenarios = [round_dir / "scenario.jsonl" for round_dir in round_dirs]
    hashes = {_sha256(path) for path in scenarios}
    if len(hashes) != 1:
        raise RuntimeError("各轮场景文件不一致，检查配置和种子")

    shutil.copy(scenarios[0], dest / "scenario.jsonl")
    for index, round_dir in enumerate(round_dirs):
        run_dir = dest / f"run_{index:02d}"
        run_dir.mkdir()
        shutil.copy(round_dir / "run_00" / "events.jsonl", run_dir / "events.jsonl")

    analyze(str(dest))


def run_ab(
    baseline_dir: str,
    opt_dir: str,
    config: str,
    rounds: int,
    out_dir: str,
    batch_interval: float,
    model_name: str = "mock-model",
) -> Dict[str, Any]:
    baseline = Path(baseline_dir).resolve()
    optimized = Path(opt_dir).resolve()
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    baseline_rounds: List[Path] = []
    opt_rounds: List[Path] = []

    for index in range(rounds):
        baseline_out = out / f"baseline_round_{index:02d}"
        opt_out = out / f"opt_round_{index:02d}"
        baseline_rounds.append(baseline_out)
        opt_rounds.append(opt_out)

        # 交替顺序，抵消运行先后带来的系统状态差异。
        if index % 2 == 0:
            print(f"[ab] 第 {index + 1} 轮：基线 -> 优化")
            _run_side(baseline, config, baseline_out, batch_interval, model_name)
            _run_side(optimized, config, opt_out, batch_interval, model_name)
        else:
            print(f"[ab] 第 {index + 1} 轮：优化 -> 基线")
            _run_side(optimized, config, opt_out, batch_interval, model_name)
            _run_side(baseline, config, baseline_out, batch_interval, model_name)

    baseline_result = out / "baseline"
    opt_result = out / "optimized"
    _collect(baseline_rounds, baseline_result)
    _collect(opt_rounds, opt_result)

    summary = compare(str(baseline_result), str(opt_result), str(out))

    meta = {
        "baseline_dir": str(baseline),
        "opt_dir": str(optimized),
        "baseline_rev": _git_rev(baseline),
        "opt_rev": _git_rev(optimized),
        "config": config,
        "rounds": rounds,
        "batch_interval": batch_interval,
        "model_name": model_name,
        "paired_requests": summary["paired_requests"],
    }
    with open(out / "ab_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    print()
    print(format_table(summary, "all"))
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="交替运行基线分支和优化分支")
    parser.add_argument("--baseline-dir", required=True, help="基线 worktree 目录")
    parser.add_argument("--opt-dir", required=True, help="优化 worktree 目录")
    parser.add_argument("--config", required=True, help="场景配置，相对路径会按各自 worktree 解析")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--out", required=True, help="输出目录")
    parser.add_argument("--batch-interval", type=float, default=0.1)
    parser.add_argument("--model-name", default="mock-model")
    args = parser.parse_args(argv)

    run_ab(
        baseline_dir=args.baseline_dir,
        opt_dir=args.opt_dir,
        config=args.config,
        rounds=args.rounds,
        out_dir=args.out,
        batch_interval=args.batch_interval,
        model_name=args.model_name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
