"""有界并发持续调度。

与原版按批次执行不同，这里维护一个等待列表和一个在途集合：

- 同时在跑的后端调用不超过 max_concurrency。
- 任何一个请求结束就立即从等待列表补充下一个。
- 新请求到达时如果有空位，直接开始执行，不等批次边界。
- 超时后先通知调用方，底层调用继续跟踪，直到真正结束才释放名额。

选择下一个请求的规则由 selection 回调决定，默认按到达顺序。回调接收等待列表
和当前单调时钟，返回要取走的下标。
"""

import logging
import threading
import time
from queue import Empty
from typing import Any, Callable, Dict, List, Optional

from cerebrum.llm.apis import LLMResponse

from aios.utils import probe

from .fifo_scheduler import FIFOScheduler

logger = logging.getLogger(__name__)


class BoundedScheduler(FIFOScheduler):
    """固定并发上限、完成即补位的调度器。"""

    def __init__(
        self,
        *args: Any,
        max_concurrency: int = 4,
        selection: Optional[Callable[[List[Any], float], int]] = None,
        predictor: Any = None,
        request_timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency 必须大于等于 1")

        super().__init__(*args, **kwargs)
        self.max_concurrency = max_concurrency
        self.selection = selection
        self.predictor = predictor
        self.request_timeout = request_timeout

        self._waiting: List[Any] = []
        self._in_flight: Dict[Any, float] = {}
        self._cv = threading.Condition()
        self._collector: Optional[threading.Thread] = None
        self._dispatcher: Optional[threading.Thread] = None
        self._collector_done = False

    def _enqueue_waiting(self, syscall: Any) -> None:
        with self._cv:
            self._waiting.append(syscall)
            self._cv.notify_all()

    def _collect_requests(self) -> None:
        """把全局队列里的请求搬进等待列表。"""
        while self.active:
            try:
                syscall = self.get_llm_syscall()
            except Empty:
                continue
            self._enqueue_waiting(syscall)

        # 关闭时把已经进入队列的请求收完，避免调用方永久等待。
        while True:
            try:
                syscall = self.get_llm_syscall()
            except Empty:
                break
            self._enqueue_waiting(syscall)

        with self._cv:
            self._collector_done = True
            self._cv.notify_all()

    def _select_index(self, now: float) -> int:
        if self.selection is None:
            return 0
        return self.selection(self._waiting, now)

    def _dispatch_requests(self) -> None:
        """补位循环。"""
        while True:
            starts: List[Any] = []
            with self._cv:
                if not self.active and self._collector_done and not self._waiting and not self._in_flight:
                    return

                now = time.monotonic()
                while self._waiting and len(self._in_flight) < self.max_concurrency:
                    index = self._select_index(now)
                    syscall = self._waiting.pop(index)
                    self._in_flight[syscall] = now
                    # 与批次调度器共用 batch 事件，表示请求被调度器取走。
                    probe.emit("batch", syscall=syscall)
                    probe.emit("admit", syscall=syscall, in_flight=len(self._in_flight))
                    if self.predictor is not None:
                        probe.emit(
                            "predict",
                            syscall=syscall,
                            predicted_ms=round(self.predictor.predict(syscall), 3),
                        )
                    starts.append(syscall)

                if not starts:
                    self._cv.wait(0.05)

            for syscall in starts:
                self._start_worker(syscall, now)

            if self.request_timeout:
                self._expire_in_flight()

    def _start_worker(self, syscall: Any, started: float) -> None:
        thread = threading.Thread(
            target=self._run_worker,
            args=(syscall, started),
            name="BoundedWorker",
            daemon=True,
        )
        thread.start()

    def _run_worker(self, syscall: Any, started: float) -> None:
        try:
            # 单请求走适配器原有路径，路由、异常分类和结果写回保持一致。
            self.llm.execute_llm_syscalls([syscall])
        except Exception as exc:  # 适配器内部已处理大多数异常，这里兜底
            logger.error("有界调度工作线程异常: %s", exc, exc_info=True)
            try:
                response = self.llm._handle_completion_error(exc, "bounded-scheduler")
                self.llm._notify_syscall(syscall, response, "error")
            except Exception:
                syscall.event.set()
        finally:
            elapsed_ms = (time.monotonic() - started) * 1000.0
            if self.predictor is not None:
                self.predictor.observe(syscall, elapsed_ms)
            with self._cv:
                self._in_flight.pop(syscall, None)
                probe.emit("release", syscall=syscall, in_flight=len(self._in_flight))
                self._cv.notify_all()

    def _expire_in_flight(self) -> None:
        """超时先通知调用方，名额继续占用到实际结束。"""
        now = time.monotonic()
        expired: List[Any] = []
        with self._cv:
            for syscall, started in self._in_flight.items():
                if getattr(syscall, "_result_notified", False):
                    continue
                if now - started >= self.request_timeout:
                    expired.append(syscall)

        for syscall in expired:
            response = LLMResponse(
                response_message=None,
                error="Request timed out in scheduler.",
                finished=True,
                status_code=408,
            )
            self.llm._notify_syscall(syscall, response, "done")

    def start(self) -> None:
        self.active = True
        self.start_processing_threads(
            [
                self.process_memory_requests,
                self.process_storage_requests,
                self.process_tool_requests,
            ]
        )

        self._collector = threading.Thread(
            target=self._collect_requests, name="BoundedCollector", daemon=True
        )
        self._dispatcher = threading.Thread(
            target=self._dispatch_requests, name="BoundedDispatcher", daemon=True
        )
        self._collector.start()
        self._dispatcher.start()

    def stop(self) -> None:
        self.active = False
        with self._cv:
            self._cv.notify_all()

        if self._collector is not None:
            self._collector.join()
        with self._cv:
            self._cv.notify_all()
        if self._dispatcher is not None:
            self._dispatcher.join()

        self.stop_processing_threads()
