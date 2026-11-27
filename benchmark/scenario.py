"""场景生成。

场景文件是实验的唯一事实来源：每个请求的到达时间、服务时间、故障类型都在
这里一次算好并写盘。不同策略跑同一份场景时，每个请求的耗时完全一致。

服务时间有两种来源：
1. 推导模式，默认。按模型固定开销、输入 token、输出 token 计算，再乘一个对数正态抖动。
2. 固定模式，在配置里给每个请求的 service_ms，主要用于正确性测试和精确对照。
"""

import copy
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:  # pragma: no cover - 仓库本身依赖 pyyaml
    yaml = None


DEFAULTS: Dict[str, Any] = {
    "name": "scenario",
    "seed": 0,
    "num_tasks": 20,
    "steps": {"min": 1, "max": 1, "gap_ms": 200.0},
    "arrival": {"pattern": "uniform", "rate_per_sec": 20.0, "jitter": 0.5},
    "burst": {"size": 10, "gap_ms": 300.0},
    "prompt": {"tokens_min": 50, "tokens_max": 1000},
    "output": {"median_tokens": 60, "sigma": 0.8, "min_tokens": 1},
    "max_tokens": 512,
    "model": {
        "base_ms": 20.0,
        "prefill_ms_per_token": 0.02,
        "decode_ms_per_token": 5.0,
        "jitter_sigma": 0.15,
    },
    "faults": {
        "error_rate": 0.0,
        "timeout_rate": 0.0,
        "slow_rate": 0.0,
        "slow_extra_ms": 3000.0,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """读取 YAML 配置并和默认值合并。"""
    data: Dict[str, Any] = {}
    if path:
        if yaml is None:
            raise RuntimeError("读取 YAML 需要 pyyaml")
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    return _deep_merge(DEFAULTS, data)


def _arrival_offsets(rng: random.Random, count: int, config: Dict[str, Any]) -> List[float]:
    """返回每个任务的到达偏移，单位毫秒。"""
    arrival = config["arrival"]
    pattern = arrival.get("pattern", "uniform")
    rate = float(arrival.get("rate_per_sec", 20.0))
    jitter = float(arrival.get("jitter", 0.5))

    if rate <= 0:
        raise ValueError("arrival.rate_per_sec 必须大于 0")

    offsets: List[float] = []
    now = 0.0

    if pattern == "all_at_once":
        return [0.0] * count

    if pattern == "uniform":
        for _ in range(count):
            offsets.append(now * 1000.0)
            factor = rng.uniform(max(0.0, 1.0 - jitter), 1.0 + jitter)
            now += factor / rate
        return offsets

    if pattern == "poisson":
        for _ in range(count):
            offsets.append(now * 1000.0)
            now += rng.expovariate(rate)
        return offsets

    if pattern == "burst":
        burst = config["burst"]
        size = max(1, int(burst.get("size", 10)))
        gap = float(burst.get("gap_ms", 300.0)) / 1000.0
        for index in range(count):
            base = (index // size) * gap
            offsets.append((base + rng.uniform(0.0, 0.01)) * 1000.0)
        return offsets

    raise ValueError(f"未知的到达模式: {pattern}")


def _sample_output_tokens(rng: random.Random, config: Dict[str, Any]) -> int:
    output = config["output"]
    median = max(1.0, float(output.get("median_tokens", 60)))
    sigma = float(output.get("sigma", 0.8))
    floor = int(output.get("min_tokens", 1))
    cap = int(config.get("max_tokens", 512))

    value = median * math.exp(sigma * rng.gauss(0.0, 1.0))
    return int(min(cap, max(floor, round(value))))


def _service_ms(rng: random.Random, config: Dict[str, Any], prompt_tokens: int, output_tokens: int) -> float:
    model = config["model"]
    base = float(model.get("base_ms", 20.0))
    prefill = float(model.get("prefill_ms_per_token", 0.02))
    decode = float(model.get("decode_ms_per_token", 5.0))
    jitter_sigma = float(model.get("jitter_sigma", 0.0))

    value = base + prefill * prompt_tokens + decode * output_tokens
    if jitter_sigma > 0:
        value *= math.exp(jitter_sigma * rng.gauss(0.0, 1.0))
    return max(1.0, value)


def _pick_fault(rng: random.Random, config: Dict[str, Any]) -> str:
    faults = config["faults"]
    error_rate = float(faults.get("error_rate", 0.0))
    timeout_rate = float(faults.get("timeout_rate", 0.0))
    slow_rate = float(faults.get("slow_rate", 0.0))

    draw = rng.random()
    if draw < error_rate:
        return "error"
    if draw < error_rate + timeout_rate:
        return "timeout"
    if draw < error_rate + timeout_rate + slow_rate:
        return "slow"
    return "ok"


def generate(config: Dict[str, Any], path: Optional[str] = None) -> List[Dict[str, Any]]:
    """生成场景并返回请求列表。给出 path 时同时写成 JSONL。"""
    rng = random.Random(int(config.get("seed", 0)))
    num_tasks = int(config.get("num_tasks", 20))
    steps_cfg = config["steps"]
    step_min = max(1, int(steps_cfg.get("min", 1)))
    step_max = max(step_min, int(steps_cfg.get("max", step_min)))
    step_gap_ms = float(steps_cfg.get("gap_ms", 0.0))

    prompt_cfg = config["prompt"]
    prompt_min = max(1, int(prompt_cfg.get("tokens_min", 50)))
    prompt_max = max(prompt_min, int(prompt_cfg.get("tokens_max", 1000)))
    max_tokens = int(config.get("max_tokens", 512))

    slow_extra_ms = float(config["faults"].get("slow_extra_ms", 3000.0))
    offsets = _arrival_offsets(rng, num_tasks, config)

    rows: List[Dict[str, Any]] = []
    for task_index in range(num_tasks):
        task_id = f"t{task_index:05d}"
        num_steps = rng.randint(step_min, step_max)
        prompt_tokens = rng.randint(prompt_min, prompt_max)

        for step_index in range(num_steps):
            if step_index > 0:
                prompt_tokens += rng.randint(20, 200)
            prompt_tokens = min(prompt_tokens, 32000)

            output_tokens = _sample_output_tokens(rng, config)
            service_ms = _service_ms(rng, config, prompt_tokens, output_tokens)
            fault = _pick_fault(rng, config)
            if fault == "slow":
                service_ms += slow_extra_ms

            rows.append(
                {
                    "request_id": f"{task_id}-s{step_index}",
                    "task_id": task_id,
                    "step_index": step_index,
                    "num_steps": num_steps,
                    "arrival_offset_ms": offsets[task_index] if step_index == 0 else None,
                    "step_gap_ms": step_gap_ms if step_index > 0 else None,
                    "prompt_tokens": prompt_tokens,
                    "max_tokens": max_tokens,
                    "output_tokens": output_tokens,
                    "service_ms": round(service_ms, 3),
                    "fault": fault,
                }
            )

    if path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    return rows


def load_scenario(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
