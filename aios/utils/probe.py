"""可选打点探针。

默认不启用，未安装接收器时 emit 只做一次函数调用，开销可以忽略。
安装接收器后，aios 内部少数位置会把事件交给接收器，用于测量调度行为。

所有时间戳使用 time.monotonic()，避免系统时间调整影响测量结果。
"""

import time
from typing import Any, Callable, Dict, Optional

_sink: Optional[Callable[[Dict[str, Any]], None]] = None


def set_sink(sink: Optional[Callable[[Dict[str, Any]], None]]) -> None:
    """安装事件接收器。传入 None 等同于关闭。"""
    global _sink
    _sink = sink


def clear_sink() -> None:
    """移除事件接收器。"""
    set_sink(None)


def is_enabled() -> bool:
    """当前是否已安装接收器。"""
    return _sink is not None


def _bench_fields(syscall: Any) -> Dict[str, Any]:
    """从 syscall 的 query 上取出实验用字段。

    只有 benchmark.query.BenchLLMQuery 才带这些字段，普通请求会被跳过。
    """
    query = getattr(syscall, "query", None)
    if query is None:
        return {}

    fields: Dict[str, Any] = {}
    for name in ("bench_request_id", "bench_task_id", "bench_step_index"):
        value = getattr(query, name, None)
        if value is not None:
            fields[name] = value
    return fields


def emit(event: str, syscall: Any = None, **fields: Any) -> None:
    """记录一个事件。

    Args:
        event: 事件名，例如 enqueue、exec_start、notify。
        syscall: 可选的 syscall 对象，带实验字段时会一并写进记录。
        **fields: 额外字段，会覆盖同名默认字段。
    """
    sink = _sink
    if sink is None:
        return

    record: Dict[str, Any] = {"event": event, "ts": time.monotonic()}
    if syscall is not None:
        record.update(_bench_fields(syscall))
    record.update(fields)
    sink(record)
