"""等待队列的选择规则。

所有规则接收等待列表和当前单调时钟，返回下一个要执行的请求下标。
同分时按到达顺序，也就是列表中的先后顺序。
"""

import time
from typing import Any, Callable, List

from .duration import DurationPredictor

Selection = Callable[[List[Any], float], int]


def fifo(waiting: List[Any], now: float) -> int:
    return 0


def _wait_seconds(syscall: Any, now_mono: float) -> float:
    """等待时长。

    请求的创建时间用的是 time.time()，这里用墙钟计算，避免混用两个时钟。
    """
    created = getattr(syscall, "created_time", None)
    if created is None:
        return 0.0
    return max(0.0, time.time() - created)


def make_shortest_first(predictor: DurationPredictor) -> Selection:
    def select(waiting: List[Any], now: float) -> int:
        best = 0
        best_value = None
        for index, syscall in enumerate(waiting):
            value = predictor.predict(syscall)
            if best_value is None or value < best_value:
                best = index
                best_value = value
        return best

    return select


def make_hrrn(predictor: DurationPredictor) -> Selection:
    """最高响应比优先。

    响应比 = 1 + 等待时长 / 预计耗时。等待越久、预计越短的请求得分越高。
    等待时长在分子里持续增长，长期等待的长请求不会一直输给新来的短请求。
    """

    def select(waiting: List[Any], now: float) -> int:
        best = 0
        best_score = None
        for index, syscall in enumerate(waiting):
            predicted = max(1.0, predictor.predict(syscall))
            wait = _wait_seconds(syscall, now)
            score = 1.0 + wait * 1000.0 / predicted
            if best_score is None or score > best_score:
                best = index
                best_score = score
        return best

    return select


def make_selection(policy: str, predictor: DurationPredictor) -> Selection:
    policy = (policy or "fifo").lower()
    if policy == "fifo":
        return fifo
    if policy in ("shortest", "shortest_first"):
        return make_shortest_first(predictor)
    if policy in ("hrrn", "response_ratio"):
        return make_hrrn(predictor)
    raise ValueError(f"未知的选择策略: {policy}")
