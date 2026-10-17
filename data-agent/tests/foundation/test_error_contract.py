from app.foundation.errors import ApplicationError, ErrorContract


def test_error_contract_exposes_only_versioned_safe_fields() -> None:
    contract = ErrorContract(
        code="query.timeout",
        message="请求处理超时。",
        retryable=True,
        details={"deadline_ms": 1000},
    )

    assert contract.to_dict() == {
        "schema_version": "error-v1",
        "code": "query.timeout",
        "message": "请求处理超时。",
        "retryable": True,
        "details": {"deadline_ms": 1000},
    }
    assert str(ApplicationError(contract)) == "请求处理超时。"
