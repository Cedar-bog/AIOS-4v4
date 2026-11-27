"""事件记录。

JsonlSink 把探针事件按行写入 JSONL 文件，写入过程加锁，可以被多个线程同时调用。
"""

import json
import threading
from typing import Any, Dict, List, Optional


class JsonlSink:
    """线程安全的 JSONL 写入器。"""

    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._fh = open(path, "w", encoding="utf-8", buffering=1)
        self._count = 0

    def __call__(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            if self._fh.closed:
                return
            self._fh.write(line + "\n")
            self._count += 1

    @property
    def count(self) -> int:
        return self._count

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()


class MemorySink:
    """把事件收在内存里，供自检和单元测试使用。"""

    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def __call__(self, record: Dict[str, Any]) -> None:
        with self._lock:
            self.records.append(record)

    def find(self, event: str, request_id: Optional[str] = None) -> List[Dict[str, Any]]:
        result = []
        for record in self.records:
            if record.get("event") != event:
                continue
            if request_id is not None and record.get("bench_request_id") != request_id:
                continue
            result.append(record)
        return result
