"""负载执行。

每个任务一个线程。单步任务就是一个任务发一个请求；多步任务按顺序发请求，
上一步返回后等待 gap_ms 再发下一步。请求本身走真实的 execute_request 路径，
调用方在阻塞返回的那一刻记录时间戳。
"""

import threading
import time
from typing import Any, Callable, Dict, List, Optional

from aios.utils import probe

from .query import BenchLLMQuery


def _build_messages(prompt_tokens: int) -> List[Dict[str, str]]:
    """构造和 token 数大致相当的假提示。"""
    # 用一个固定单词重复，长度接近 4 字符每 token 的粗略估计。
    body = ("token " * max(1, prompt_tokens)).strip()
    return [
        {"role": "system", "content": "You are a benchmark agent."},
        {"role": "user", "content": body},
    ]


def _response_status(payload: Any) -> Dict[str, Any]:
    """从 execute_request 的返回值里取出状态码和错误信息。"""
    if not isinstance(payload, dict):
        return {"status_code": None, "error": "unexpected payload"}

    response = payload.get("response")
    if response is None:
        return {"status_code": None, "error": "missing response"}

    return {
        "status_code": getattr(response, "status_code", None),
        "error": getattr(response, "error", None),
    }


def _run_task(
    rows: List[Dict[str, Any]],
    execute_request: Callable[[str, Any], Any],
    start_mono: float,
    results: List[Dict[str, Any]],
    lock: threading.Lock,
) -> None:
    previous_return: Optional[float] = None

    for row in rows:
        if row["step_index"] == 0:
            target = start_mono + float(row["arrival_offset_ms"]) / 1000.0
        else:
            gap = float(row.get("step_gap_ms") or 0.0) / 1000.0
            target = (previous_return or time.monotonic()) + gap

        delay = target - time.monotonic()
        if delay > 0:
            time.sleep(delay)

        request_id = row["request_id"]
        task_id = row["task_id"]
        step_index = row["step_index"]

        messages = _build_messages(int(row["prompt_tokens"]))
        query = BenchLLMQuery(
            messages=messages,
            max_new_tokens=int(row["max_tokens"]),
            action_type="chat",
            bench_request_id=request_id,
            bench_task_id=task_id,
            bench_step_index=step_index,
            bench_service_ms=float(row["service_ms"]),
            bench_fault=row.get("fault") or "ok",
            bench_output_tokens=int(row.get("output_tokens") or 0),
        )

        probe.emit(
            "arrive",
            bench_request_id=request_id,
            bench_task_id=task_id,
            bench_step_index=step_index,
        )

        started = time.monotonic()
        status: Dict[str, Any]
        try:
            payload = execute_request(f"agent_{task_id}", query)
            status = _response_status(payload)
        except Exception as exc:  # 调用方路径异常，记录后停止这条链
            status = {"status_code": None, "error": repr(exc)}
            payload = None

        finished = time.monotonic()
        probe.emit(
            "return",
            bench_request_id=request_id,
            bench_task_id=task_id,
            bench_step_index=step_index,
            status_code=status["status_code"],
            error=status["error"],
        )

        with lock:
            results.append(
                {
                    "request_id": request_id,
                    "task_id": task_id,
                    "step_index": step_index,
                    "arrive": started,
                    "return": finished,
                    "status_code": status["status_code"],
                    "error": status["error"],
                    "has_payload": payload is not None,
                }
            )

        previous_return = finished

        if payload is None:
            break


def run_scenario(
    rows: List[Dict[str, Any]],
    execute_request: Callable[[str, Any], Any],
    start_mono: Optional[float] = None,
) -> Dict[str, Any]:
    """按场景并发执行所有任务，返回汇总信息。"""
    if start_mono is None:
        start_mono = time.monotonic()

    tasks: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        tasks.setdefault(row["task_id"], []).append(row)

    for task_rows in tasks.values():
        task_rows.sort(key=lambda item: item["step_index"])

    results: List[Dict[str, Any]] = []
    lock = threading.Lock()
    threads = [
        threading.Thread(
            target=_run_task,
            args=(task_rows, execute_request, start_mono, results, lock),
            name=f"bench-{task_id}",
            daemon=True,
        )
        for task_id, task_rows in tasks.items()
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    delivered = sum(1 for item in results if item["has_payload"])
    failed = [item for item in results if not item["has_payload"]]

    return {
        "tasks": len(tasks),
        "requests": len(results),
        "delivered": delivered,
        "failed": len(failed),
        "failures": failed,
        "wall_ms": (time.monotonic() - start_mono) * 1000.0,
    }
