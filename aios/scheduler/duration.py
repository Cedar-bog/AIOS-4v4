"""请求耗时估计。

按模型、输入长度和输出上限分组，用已完成请求的服务时长滑动均值估计。
冷启动阶段使用配置的固定值，之后只使用真实完成的请求更新估计。

输入长度用字符数除以 4 粗估，不依赖具体分词器。估计器只读取请求本身的字段，
不读取场景文件里的真实服务时间，避免把答案泄露给调度器。
"""

import threading
from collections import defaultdict, deque
from typing import Any, Deque, Dict, Tuple


class DurationPredictor:
    """按请求特征分组的滑动均值估计器。"""

    def __init__(
        self,
        model_name: str = "default",
        window: int = 32,
        cold_start_ms: float = 500.0,
        token_bucket: int = 128,
    ) -> None:
        self.model_name = model_name
        self.window = max(1, int(window))
        self.cold_start_ms = max(1.0, float(cold_start_ms))
        self.token_bucket = max(1, int(token_bucket))

        self._lock = threading.Lock()
        self._samples: Dict[Tuple[str, int, int], Deque[float]] = defaultdict(
            lambda: deque(maxlen=self.window)
        )
        self._global: Deque[float] = deque(maxlen=self.window * 4)

    @staticmethod
    def _prompt_tokens(syscall: Any) -> int:
        query = getattr(syscall, "query", None)
        messages = getattr(query, "messages", None) or []
        chars = 0
        for message in messages:
            chars += len(str(message.get("content", "")))
        return max(1, chars // 4)

    def key(self, syscall: Any) -> Tuple[str, int, int]:
        query = getattr(syscall, "query", None)
        prompt_tokens = self._prompt_tokens(syscall)
        max_tokens = int(getattr(query, "max_new_tokens", 0) or 0)
        bucket = prompt_tokens // self.token_bucket
        return self.model_name, bucket, max_tokens

    def predict(self, syscall: Any) -> float:
        key = self.key(syscall)
        with self._lock:
            samples = self._samples.get(key)
            if samples:
                return sum(samples) / len(samples)
            if self._global:
                return sum(self._global) / len(self._global)
        return self.cold_start_ms

    def observe(self, syscall: Any, service_ms: float) -> None:
        value = max(1.0, float(service_ms))
        key = self.key(syscall)
        with self._lock:
            self._samples[key].append(value)
            self._global.append(value)

    def sample_count(self) -> int:
        with self._lock:
            return len(self._global)
