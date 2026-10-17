import json

import pytest

import evals.legacy.recheck_run as recheck_module


@pytest.mark.asyncio
async def test_recheck_uses_question_order_semantics_and_preserves_audit_fields(
    tmp_path, monkeypatch,
):
    run_path = tmp_path / "eval" / "runs" / "run_synthetic.json"
    run_path.parent.mkdir(parents=True)
    expected_sql = (
        "SELECT kind, COUNT(*) AS n FROM events GROUP BY kind ORDER BY n DESC"
    )
    generated_sql = "SELECT kind, COUNT(*) AS n FROM events GROUP BY kind"
    payload = {
        "valid": True,
        "run_id": "synthetic-order-contract",
        "dataset_suite": "synthetic_suite",
        "aggregate": {"result_equivalent": 0},
        "cases": [
            {
                "id": "ordinary-report",
                "question": "旧制品中的题面不会覆盖当前合同",
                "expected_sql": expected_sql,
                "gen_sql": generated_sql,
                "result_ordered": True,
                "result_equivalent": False,
            },
            {
                "id": "explicit-sort",
                "question": "旧制品中的题面同样不是合同真值",
                "expected_sql": expected_sql,
                "gen_sql": generated_sql,
                "result_ordered": False,
                "result_equivalent": True,
            },
            {
                "id": "minimum-projection",
                "question": "旧制品中的投影题面同样不是合同真值",
                "expected_sql": (
                    "SELECT product_name, category, brand FROM dim_product "
                    "ORDER BY product_name"
                ),
                "gen_sql": (
                    "SELECT product_name FROM dim_product ORDER BY product_name"
                ),
                "result_ordered": True,
                "result_equivalent": False,
            },
            {
                "id": "generated-exec-error",
                "question": "旧制品题面",
                "expected_sql": "SELECT kind, COUNT(*) AS n FROM events GROUP BY kind",
                "gen_sql": "SELECT * FROM missing_relation",
                "result_ordered": False,
                "result_equivalent": False,
            },
        ],
    }
    run_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    contracts = [
        {
            "id": "ordinary-report",
            "question": "汇总各类型的记录数",
            "result_ordered": True,
        },
        {
            "id": "explicit-sort",
            "question": "按记录数降序排列各类型",
            "result_ordered": False,
        },
        {
            "id": "minimum-projection",
            "question": "按商品名称升序列出商品目录中的商品",
            "result_ordered": True,
        },
        {
            "id": "generated-exec-error",
            "question": "汇总各类型记录数",
            "result_ordered": False,
        },
    ]

    class FakeSession:
        def __init__(self):
            self.rollbacks = 0

        async def rollback(self):
            self.rollbacks += 1

    class FakeSessionContext:
        def __init__(self, session):
            self.session = session

        async def __aenter__(self):
            return self.session

        async def __aexit__(self, *_):
            return False

    class FakeManager:
        def __init__(self):
            self.session = FakeSession()
            self.closed = False

        def init(self):
            return None

        def session_factory(self):
            return FakeSessionContext(self.session)

        async def close(self):
            self.closed = True

    class FakeRepo:
        def __init__(self, _session):
            pass

        async def execute_sql_for_evaluation(self, sql):
            if "missing_relation" in sql:
                raise RuntimeError(1146, "raw database detail must not persist")
            if "product_name, category, brand" in sql:
                return [{"product_name": "A", "category": "C", "brand": "B"}]
            if "product_name" in sql:
                return [{"product_name": "A"}]
            rows = [{"kind": "a", "n": 2}, {"kind": "b", "n": 1}]
            return rows if "ORDER BY" in sql else list(reversed(rows))

    manager = FakeManager()
    monkeypatch.setattr(recheck_module, "dw_mysql_client_manager", manager)
    monkeypatch.setattr(recheck_module, "DWMysqlRepository", FakeRepo)
    monkeypatch.setattr(
        recheck_module,
        "SUITE_SPECS",
        {"synthetic_suite": {"promotion_status": "draft"}},
    )
    monkeypatch.setattr(recheck_module, "load_cases", lambda *, suite: contracts)
    monkeypatch.setattr(
        recheck_module,
        "aggregate",
        lambda cases: {
            "n": len(cases),
            "result_equivalent": sum(case["result_equivalent"] for case in cases),
            "latency_p95": 0,
        },
    )
    monkeypatch.setattr(recheck_module, "aggregate_by_category", lambda _cases: {})
    monkeypatch.setattr(recheck_module, "render_report", lambda *_: "synthetic report")

    result = await recheck_module.recheck(run_path)

    ordinary, explicit, projected, generated_error = result["cases"]
    assert ordinary["result_equivalent"] is True
    assert ordinary["result_ordered"] is False
    assert ordinary["result_ordered_declared"] is True
    assert ordinary["result_equivalent_before_contract_v9"] is False
    assert ordinary["result_ordered_before_contract_v9"] is True

    assert explicit["result_equivalent"] is False
    assert explicit["result_ordered"] is True
    assert explicit["result_ordered_declared"] is False
    assert projected["result_equivalent"] is True
    assert (
        projected["result_equivalence_evidence"]["comparison_mode"]
        == "minimum_sufficient_projection"
    )
    assert generated_error["result_equivalent"] is False
    assert generated_error["recheck_execution"] == {
        "status": "generated_execution_error",
        "error_type": "RuntimeError",
        "db_error_code": 1146,
    }
    assert "raw database detail" not in json.dumps(result, ensure_ascii=False)
    assert result["evaluation_contract_version"] == 9
    assert result["recheck_history"][-1]["contract_version"] == 9
    assert manager.closed is True
    assert (
        tmp_path
        / "eval"
        / "reports"
        / "nl2sql"
        / "report_synthetic-order-contract_contract_v9.md"
    ).exists()
