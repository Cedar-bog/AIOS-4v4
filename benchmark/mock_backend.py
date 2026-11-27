"""假 LLM 后端。

只替换最底层的一次模型调用，调度器、队列、线程池、syscall 通知全部走真实代码。
服务时间由场景文件预先算好，写进 BenchLLMQuery.bench_service_ms，后端照做，
这样不同策略跑同一份场景时每个请求的耗时完全一致。
"""

import time
from typing import Any, Dict, List, Optional, Tuple, Union

from aios.llm_core.adapter import LLMAdapter, LLMConfig
from aios.utils import probe


class MockLLMAdapter(LLMAdapter):
    """不联网的 LLMAdapter，按请求给定的毫秒数睡眠。"""

    def __init__(self, model_name: str = "mock-model", log_mode: str = "silent") -> None:
        self._mock_model_name = model_name
        super().__init__(
            llm_configs=[{"name": model_name, "backend": "mock"}],
            log_mode=log_mode,
        )

    def _initialize_llms(self) -> None:
        """跳过真实后端初始化，只登记一个占位模型。"""
        self.llm_configs = [LLMConfig(name=self._mock_model_name, backend="mock")]
        self.llms = [self._mock_model_name]
        self.available_llm_names = [self._mock_model_name]

    def _setup_api_keys(self) -> None:
        """实验不访问真实服务，跳过密钥检查，避免无谓的控制台输出。"""
        return None

    def _get_model_response(
        self,
        model_name: str,
        model: Union[str, Any],
        messages: List[Dict],
        tools: Optional[List],
        llm_syscall,
        api_base: Optional[str] = None,
        message_return_type: Optional[str] = "text",
        response_format: Optional[Dict] = None,
        temperature: float = 1.0,
        max_tokens: int = 1000,
    ) -> Tuple[Union[str, List, Dict], bool]:
        query = llm_syscall.query

        request_id = getattr(query, "bench_request_id", None) or "unknown"
        service_ms = getattr(query, "bench_service_ms", None)
        fault = getattr(query, "bench_fault", None) or "ok"
        output_tokens = getattr(query, "bench_output_tokens", 0)

        if service_ms is None:
            # 没有实验字段时给一个保守的默认值，方便手工调用。
            service_ms = 10.0

        probe.emit("backend_start", syscall=llm_syscall, fault=fault)
        time.sleep(max(0.0, float(service_ms)) / 1000.0)
        probe.emit("backend_done", syscall=llm_syscall, fault=fault)

        if fault == "error":
            raise RuntimeError(f"simulated backend error for {request_id}")
        if fault == "timeout":
            # 消息里带 timeout，adapter 会归类为 408。
            raise TimeoutError(f"simulated timeout for {request_id}")

        content = f"[mock:{request_id}] output_tokens={output_tokens}"
        return content, True
