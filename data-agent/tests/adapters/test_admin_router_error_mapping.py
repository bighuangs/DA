"""Admin Router 对 Application typed errors 的状态映射。"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.adapters.inbound.http.routers.admin_router import _call
from app.application.services.meta_admin_service import (
    AdminConflictError,
    AdminForbiddenError,
    AdminNotFoundError,
    AdminValidationError,
)


async def _fail(error):
    raise error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "status"),
    [
        (AdminConflictError("冲突"), 409),
        (AdminNotFoundError("不存在"), 404),
        (AdminValidationError("无效"), 422),
        (AdminForbiddenError("无权"), 403),
    ],
)
async def test_typed_error_mapping_preserves_status_and_detail(error, status) -> None:
    with pytest.raises(HTTPException) as captured:
        await _call(_fail(error))

    assert captured.value.status_code == status
    assert captured.value.detail == str(error)
