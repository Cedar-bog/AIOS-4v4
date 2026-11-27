"""基准测试用到的请求类型。

LLMQuery 本身没有实验字段，这里用子类补上，避免改动上游数据结构。
普通路径只读取 messages、max_new_tokens 等原有字段，多余字段不影响执行。
"""

from typing import Optional

from cerebrum.llm.apis import LLMQuery


class BenchLLMQuery(LLMQuery):
    """带实验标记的 LLM 请求。

    bench_request_id: 场景文件里的请求编号。
    bench_task_id: 所属任务编号，单步请求时与请求编号同源。
    bench_step_index: 任务中的步骤序号，从 0 开始。
    bench_service_ms: 假后端应该消耗的服务时间，单位毫秒。
    bench_fault: 故障类型，取 ok、error、timeout、slow。
    """

    bench_request_id: Optional[str] = None
    bench_task_id: Optional[str] = None
    bench_step_index: Optional[int] = None
    bench_service_ms: Optional[float] = None
    bench_fault: Optional[str] = None
    bench_output_tokens: Optional[int] = None
