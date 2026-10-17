"""Query HTTP Adapter 的公开状态码与 SSE 合同回归测试。"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.adapters.inbound.http.dependencies import (
    get_current_principal,
    get_query_capacity,
    get_query_service,
)
from app.adapters.inbound.http.routers.query_router import query_router
from app.application.ports.http_runtime import QueryCapacityExceeded
from app.application.services.auth_service import Principal


class _Lease:
    def __init__(self) -> None:
        self.entered = False
        self.exited = False

    async def __aenter__(self) -> None:
        self.entered = True

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        self.exited = True


class _Capacity:
    def __init__(self, lease: _Lease) -> None:
        self.lease = lease
        self.workspace_ids: list[str] = []

    async def acquire(self, workspace_id: str) -> _Lease:
        self.workspace_ids.append(workspace_id)
        return self.lease


class _RejectedCapacity:
    async def acquire(self, workspace_id: str) -> _Lease:
        raise QueryCapacityExceeded(f"{workspace_id} busy")


class _QueryService:
    def __init__(self) -> None:
        self.search_kwargs: dict[str, object] | None = None

    def search(self, **kwargs: object):
        self.search_kwargs = kwargs

        async def stream():
            yield 'data: {"stage":"thinking"}\n\n'
            yield "data: [DONE]\n\n"

        return stream()


def _principal() -> Principal:
    return Principal(
        user_id="user-1",
        username="alice",
        display_name="Alice",
        workspace_id="workspace-1",
        workspace_name="Workspace",
        role="member",
    )


def _test_app(*, service: object, capacity: object) -> FastAPI:
    app = FastAPI()
    app.include_router(query_router)
    app.dependency_overrides[get_query_service] = lambda: service
    app.dependency_overrides[get_current_principal] = _principal
    app.dependency_overrides[get_query_capacity] = lambda: capacity
    return app


def test_query_preserves_sse_stream_and_holds_capacity_for_stream_lifetime() -> None:
    service = _QueryService()
    lease = _Lease()
    capacity = _Capacity(lease)

    with TestClient(_test_app(service=service, capacity=capacity)) as client:
        response = client.post(
            "/api/query",
            json={
                "query": "本月销售额",
                "chat_id": "chat-1",
                "dataset_ids": ["dataset-1"],
                "enable_analysis": True,
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text == 'data: {"stage":"thinking"}\n\ndata: [DONE]\n\n'
    assert capacity.workspace_ids == ["workspace-1"]
    assert lease.entered and lease.exited
    assert service.search_kwargs == {
        "query": "本月销售额",
        "chat_id": "chat-1",
        "dataset_id": None,
        "dataset_ids": ["dataset-1"],
        "compare_dw": False,
        "source_mode": None,
        "comparison_intent": None,
        "mapping_confirmation_id": None,
        "data_context_revision": None,
        "enable_analysis": True,
    }


def test_query_capacity_rejection_preserves_429_contract() -> None:
    with TestClient(
        _test_app(service=_QueryService(), capacity=_RejectedCapacity())
    ) as client:
        response = client.post("/api/query", json={"query": "本月销售额"})

    assert response.status_code == 429
    assert response.headers["retry-after"] == "2"
    assert response.json() == {"detail": "workspace-1 busy"}
