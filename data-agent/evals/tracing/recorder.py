"""不采集隐藏思维链的内存 Trace Recorder。"""

from __future__ import annotations

from threading import RLock
from typing import Mapping, Protocol

from evals.contracts import ContractError, EventRecord, TraceKind, TraceRecord


class TraceSink(Protocol):
    """被评目标写入可观测事件的最小协议。"""

    evaluation_id: str
    case_id: str
    trace_id: str

    def emit(self, event: EventRecord) -> None:
        """写入一个已完成或运行中的结构化事件。"""


class InMemoryTraceRecorder:
    """线程安全地收集单个 Case 的结构化事件。"""

    def __init__(self, *, evaluation_id: str, case_id: str, trace_id: str) -> None:
        self.evaluation_id = evaluation_id
        self.case_id = case_id
        self.trace_id = trace_id
        self._events: list[EventRecord] = []
        self._event_ids: set[str] = set()
        self._lock = RLock()

    def emit(self, event: EventRecord) -> None:
        """拒绝跨 Case、跨 Trace 或重复 ID 的事件。"""

        if (
            event.evaluation_id != self.evaluation_id
            or event.case_id != self.case_id
            or event.trace_id != self.trace_id
        ):
            raise ContractError("事件不能写入其他 evaluation/case/trace 的 Recorder")
        with self._lock:
            if event.event_id in self._event_ids:
                raise ContractError(f"重复 event_id: {event.event_id}")
            self._events.append(event)
            self._event_ids.add(event.event_id)

    def snapshot(
        self,
        *,
        kind: TraceKind = TraceKind.SYSTEM,
        versions: Mapping[str, str] | None = None,
    ) -> TraceRecord:
        """生成不可变 Trace 快照并执行父子引用校验。"""

        with self._lock:
            events = tuple(self._events)
        return TraceRecord(
            evaluation_id=self.evaluation_id,
            case_id=self.case_id,
            trace_id=self.trace_id,
            kind=kind,
            events=events,
            versions=versions or {},
        )
