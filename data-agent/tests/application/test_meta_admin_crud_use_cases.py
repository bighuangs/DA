"""Meta Admin CRUD、typed errors 与缓存编排回归。"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

from app.application.services.meta_admin_service import (
    AdminConflictError,
    AdminForbiddenError,
    AdminNotFoundError,
    AdminValidationError,
    MetaAdminService,
)
from app.domain.privacy.masking import MaskingPolicy


class _Cache:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def invalidate_retrieval(self) -> None:
        self.events.append("invalidate_retrieval")

    def clear_query_results(self) -> None:
        self.events.append("clear_query_results")


class _Repository:
    workspace_id = "workspace-a"

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.metrics: dict[str, object] = {}
        self.training: dict[int, object] = {}
        self.masking: dict[int, object] = {}
        self._next_training = 1
        self._next_masking = 1

    async def ensure_admin_tables(self) -> None:
        self.events.append("ensure")

    async def list_metric_infos(self, kind=None, search=None):
        del kind, search
        return list(self.metrics.values())

    async def get_metric_info(self, metric_id):
        return self.metrics.get(metric_id)

    async def create_metric_info(self, **values):
        item = SimpleNamespace(id=values.pop("metric_id"), **values)
        self.metrics[item.id] = item
        self.events.append("create_metric")
        return item

    async def update_metric_info(self, item, **values):
        for key, value in values.items():
            if key == "formula_rule" and value is None:
                continue
            setattr(item, key, value)
        self.events.append("update_metric")
        return item

    async def delete_metric_info(self, item) -> None:
        self.metrics.pop(item.id)
        self.events.append("delete_metric")

    async def list_training_infos(self, search=None):
        del search
        return list(self.training.values())

    async def get_training_info(self, training_id):
        return self.training.get(training_id)

    async def create_training_info(self, **values):
        item = SimpleNamespace(
            id=self._next_training,
            created_at=datetime(2026, 1, 1),
            **values,
        )
        self._next_training += 1
        self.training[item.id] = item
        self.events.append("create_training")
        return item

    async def update_training_info(self, item, **values):
        for key, value in values.items():
            setattr(item, key, value)
        self.events.append("update_training")
        return item

    async def delete_training_info(self, item) -> None:
        self.training.pop(item.id)
        self.events.append("delete_training")

    async def list_table_infos(self):
        return []

    async def list_masking_rules(self):
        return list(self.masking.values())

    async def get_masking_rule(self, rule_id):
        return self.masking.get(rule_id)

    async def create_masking_rule(self, **values):
        item = SimpleNamespace(
            id=self._next_masking,
            created_at=datetime(2026, 1, 1),
            **values,
        )
        self._next_masking += 1
        self.masking[item.id] = item
        self.events.append("create_masking")
        return item

    async def update_masking_rule(self, item, **values):
        for key, value in values.items():
            setattr(item, key, value)
        self.events.append("update_masking")
        return item

    async def delete_masking_rule(self, item) -> None:
        self.masking.pop(item.id)
        self.events.append("delete_masking")

    async def list_schema_sync_logs(self, limit=10):
        del limit
        return []


class _AccessPolicyAdmin:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.items: dict[int, object] = {}
        self._next_id = 1

    async def list_all(self):
        return list(self.items.values())

    async def get(self, policy_id):
        return self.items.get(policy_id)

    async def create(self, **values):
        if values["table_pattern"] == "invalid":
            raise ValueError("表匹配规则未命中当前 Schema：invalid")
        item = SimpleNamespace(
            id=self._next_id,
            workspace_id="workspace-a",
            created_at=datetime(2026, 1, 1),
            **values,
        )
        self._next_id += 1
        self.items[item.id] = item
        self.events.append("create_access")
        return item

    async def update(self, item, **values):
        for key, value in values.items():
            setattr(item, key, value)
        self.events.append("update_access")
        return item

    async def delete(self, item) -> None:
        self.items.pop(item.id)
        self.events.append("delete_access")


def _service(events: list[str]) -> MetaAdminService:
    return MetaAdminService(
        meta_repo=_Repository(events),
        projection_index=SimpleNamespace(),
        cache_invalidator=_Cache(events),
        access_policy_admin=_AccessPolicyAdmin(events),
        masking_policy=MaskingPolicy(),
    )


@pytest.mark.asyncio
async def test_all_admin_writes_own_prepare_and_cache_invalidation() -> None:
    events: list[str] = []
    service = _service(events)

    metric = await service.create_metric(
        name="GMV",
        description="成交金额",
        relevant_columns=["fact_order.amount"],
        alias=["销售额"],
        enabled=True,
        formula_rule=None,
    )
    await service.update_metric(
        metric["id"],
        description="成交总额",
        relevant_columns=["fact_order.amount"],
        alias=["GMV"],
        enabled=True,
        formula_rule=None,
    )
    await service.delete_metric(metric["id"])

    training = await service.create_training(
        question="查询 GMV",
        sql="SELECT SUM(amount) FROM fact_order",
        enabled=True,
    )
    await service.update_training(
        training["id"],
        question="查询成交额",
        sql="SELECT SUM(amount) FROM fact_order",
        enabled=False,
    )
    await service.delete_training(training["id"])

    masking = await service.create_masking_rule(
        column_pattern="*phone*",
        data_kind="phone",
        strategy="mask",
        enabled=True,
        priority=10,
        description="手机号",
    )
    await service.update_masking_rule(
        masking["id"],
        column_pattern="*mobile*",
        data_kind="phone",
        strategy="keep_last4",
        enabled=True,
        priority=20,
        description="手机号",
    )
    await service.delete_masking_rule(masking["id"])

    policy_values = {
        "role": "viewer",
        "user_id": None,
        "table_pattern": "fact_*",
        "allowed_columns": None,
        "row_filter": None,
        "access": True,
        "enabled": True,
        "priority": 100,
        "description": "只读",
    }
    policy = await service.create_access_policy(**policy_values)
    await service.update_access_policy(policy["id"], **policy_values)
    await service.delete_access_policy(policy["id"])

    assert events == [
        "ensure",
        "create_metric",
        "invalidate_retrieval",
        "ensure",
        "update_metric",
        "invalidate_retrieval",
        "ensure",
        "delete_metric",
        "invalidate_retrieval",
        "ensure",
        "create_training",
        "invalidate_retrieval",
        "ensure",
        "update_training",
        "invalidate_retrieval",
        "ensure",
        "delete_training",
        "invalidate_retrieval",
        "ensure",
        "create_masking",
        "clear_query_results",
        "ensure",
        "update_masking",
        "clear_query_results",
        "ensure",
        "delete_masking",
        "clear_query_results",
        "ensure",
        "create_access",
        "invalidate_retrieval",
        "ensure",
        "update_access",
        "invalidate_retrieval",
        "ensure",
        "delete_access",
        "invalidate_retrieval",
    ]


@pytest.mark.asyncio
async def test_admin_service_exposes_typed_business_errors() -> None:
    events: list[str] = []
    service = _service(events)
    values = {
        "name": "GMV",
        "description": "",
        "relevant_columns": [],
        "alias": [],
        "enabled": True,
        "formula_rule": None,
    }
    await service.create_metric(**values)

    with pytest.raises(AdminConflictError, match="指标名称已存在"):
        await service.create_metric(**values)
    with pytest.raises(AdminNotFoundError, match="训练样例不存在"):
        await service.delete_training(999)
    with pytest.raises(AdminValidationError, match="未命中当前 Schema"):
        await service.create_access_policy(
            role="viewer",
            user_id=None,
            table_pattern="invalid",
            allowed_columns=None,
            row_filter=None,
            access=True,
            enabled=True,
            priority=100,
            description="",
        )

    assert events.count("invalidate_retrieval") == 1


@pytest.mark.asyncio
async def test_value_search_permission_is_an_application_error() -> None:
    events: list[str] = []
    repository = _Repository(events)
    table = SimpleNamespace(id="fact_order", name="fact_order")
    column = SimpleNamespace(id="fact_order.amount", name="amount")

    async def list_tables():
        return [(table, [column])]

    repository.list_table_infos = list_tables
    service = MetaAdminService(
        meta_repo=repository,
        projection_index=SimpleNamespace(),
        cache_invalidator=_Cache(events),
        value_index=SimpleNamespace(),
        access_policy=SimpleNamespace(
            is_table_allowed=lambda table_name: table_name != "fact_order",
            is_column_allowed=lambda table_name, column_name: False,
        ),
    )

    with pytest.raises(AdminForbiddenError, match="无权检索该表"):
        await service.search_values(
            "100",
            query_mode="exact",
            table_id="fact_order",
            column_id=None,
            page=1,
            page_size=20,
        )


@pytest.mark.asyncio
async def test_status_and_rollback_own_admin_table_preparation() -> None:
    events: list[str] = []

    class _Projection:
        async def status(self, **values):
            del values
            events.append("status")
            return {"ready": True}

    class _ValueIndex:
        async def rollback(self, **values):
            events.append("rollback")
            return values

    service = MetaAdminService(
        meta_repo=_Repository(events),
        projection_index=_Projection(),
        value_index=_ValueIndex(),
        cache_invalidator=_Cache(events),
    )

    assert await service.hybrid_status() == {"ready": True}
    assert (await service.rollback_value_index("value-v2"))["physical_index"] == (
        "value-v2"
    )
    assert events == ["ensure", "status", "ensure", "rollback"]
