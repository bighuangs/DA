import asyncio
import csv
import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

import evals.integrations.ragas_analysis as ragas_analysis
from evals.integrations.ragas_analysis import (
    IdMetricScores,
    SqlJudgeSample,
    SqlMetricScore,
    build_retrieval_id_samples,
    deterministic_id_precision,
    deterministic_id_recall,
    extract_schema_ddl,
    run_ragas_analysis,
    score_retrieval_samples,
    score_sql_samples,
    stratified_sample_sql_cases,
    summarize_full_sql_runs,
    summarize_retrieval_results,
    summarize_sql_judge_results,
    write_analysis_artifacts,
)


def _hashed_value_id(workspace_id: str, column_id: str, value: str) -> str:
    digest = hashlib.sha256(
        "\x1f".join((workspace_id, column_id, value.casefold())).encode("utf-8")
    ).hexdigest()
    return f"value:{digest}"


def _retrieval_fixture() -> tuple[list[dict], list[dict], dict]:
    cases = [
        {"case_id": "RAG-001", "query": "查询一"},
        {"case_id": "RAG-002", "query": "查询二"},
    ]
    qrels = [
        {"case_id": "RAG-001", "document_id": "table:fact", "relevance_grade": "3"},
        {"case_id": "RAG-001", "document_id": "column:fact.amount", "relevance_grade": "3"},
        {"case_id": "RAG-001", "document_id": "metric:GMV", "relevance_grade": "3"},
        {"case_id": "RAG-001", "document_id": "training:成交金额", "relevance_grade": "3"},
        {
            "case_id": "RAG-001",
            "document_id": "value:dim_region.region_name:华东",
            "relevance_grade": "3",
        },
        {"case_id": "RAG-002", "document_id": "table:dim_date", "relevance_grade": "3"},
        {"case_id": "RAG-002", "document_id": "metric:忽略", "relevance_grade": "0"},
    ]
    live_run = {
        "workspace_id": "workspace-a",
        "cases": [
            {
                "rag_case_id": "RAG-001",
                "question": "查询一",
                "stage_evidence": {
                    "filtered_tables": ["fact", "dim_region", "fact"],
                    "context_columns": ["fact.amount", "fact.extra"],
                    "retrieval": {
                        "columns": [
                            {"id": "fact.noise", "rank": 2},
                            {"id": "fact.amount", "rank": 1},
                            {"id": "fact.amount", "rank": 3},
                        ],
                        "metrics": [{"id": "opaque-id", "name": "GMV", "rank": 1}],
                        "training": [
                            {"id": "42", "question": "成交金额", "rank": 1}
                        ],
                        "values": [
                            {
                                "id": _hashed_value_id(
                                    "workspace-a", "dim_region.region_name", "华东"
                                ),
                                "rank": 1,
                            }
                        ],
                    },
                },
            },
            {
                "rag_case_id": "RAG-002",
                "question": "查询二",
                "stage_evidence": {
                    "filtered_tables": ["dim_date"],
                    "context_columns": [],
                    "retrieval": {"columns": [], "metrics": [], "training": [], "values": []},
                },
            },
        ],
    }
    return cases, qrels, live_run


def test_build_retrieval_samples_keeps_route_ranks_and_normalizes_ids() -> None:
    cases, qrels, live_run = _retrieval_fixture()

    samples = build_retrieval_id_samples(cases, qrels, live_run)
    by_key = {(sample.case_id, sample.route): sample for sample in samples}

    assert [sample.route for sample in samples if sample.case_id == "RAG-001"] == [
        "table",
        "column_direct",
        "column_effective",
        "metric",
        "training",
        "value",
    ]
    assert by_key[("RAG-001", "table")].retrieved_ids == (
        "table:fact",
        "table:dim_region",
    )
    assert by_key[("RAG-001", "column_direct")].retrieved_ids == (
        "column:fact.amount",
        "column:fact.noise",
    )
    assert by_key[("RAG-001", "column_effective")].retrieved_ids == (
        "column:fact.amount",
        "column:fact.noise",
        "column:fact.extra",
    )
    assert by_key[("RAG-001", "metric")].retrieved_ids == ("metric:GMV",)
    assert by_key[("RAG-001", "training")].retrieved_ids == ("training:成交金额",)
    assert by_key[("RAG-001", "value")].retrieved_ids == (
        "value:dim_region.region_name:华东",
    )
    assert ("RAG-002", "metric") not in by_key
    assert by_key[("RAG-002", "table")].retrieved_ids == ("table:dim_date",)


def test_deterministic_id_metrics_deduplicate_and_reject_empty_qrels() -> None:
    assert deterministic_id_precision(["a"], ["a", "a", "b"]) == 0.5
    assert deterministic_id_recall(["a", "b"], ["a", "a"]) == 0.5
    assert deterministic_id_precision(["a"], []) == 0.0
    with pytest.raises(ValueError, match="正 qrel"):
        deterministic_id_recall([], ["a"])


def test_retrieval_scorer_preserves_every_sample_and_error() -> None:
    cases, qrels, live_run = _retrieval_fixture()
    samples = build_retrieval_id_samples(cases, qrels, live_run)

    async def fake_scorer(sample):
        if sample.route == "metric":
            raise RuntimeError("judge unavailable")
        return IdMetricScores(
            deterministic_id_precision(sample.reference_ids, sample.retrieved_ids),
            deterministic_id_recall(sample.reference_ids, sample.retrieved_ids),
        )

    results = asyncio.run(score_retrieval_samples(samples, fake_scorer))
    summary = summarize_retrieval_results(results)

    assert len(results) == len(samples)
    metric_result = next(row for row in results if row.route == "metric")
    assert metric_result.status == "error"
    assert metric_result.error_type == "RuntimeError"
    assert metric_result.deterministic_recall == 1.0
    assert summary["routes"]["metric"]["error_count"] == 1
    assert summary["routes"]["column_direct"]["deterministic"] == {
        "precision_macro": 0.5,
        "recall_macro": 1.0,
    }


def _sql_run(difficulty: str, positive: int, negative: int) -> dict:
    cases = []
    for label, count in ((True, positive), (False, negative)):
        for index in range(count):
            cases.append(
                {
                    "id": f"{difficulty}-{'p' if label else 'n'}-{index:02d}",
                    "difficulty": difficulty,
                    "question": f"{difficulty} question {label} {index}",
                    "gen_sql": f"SELECT {index}",
                    "expected_sql": f"SELECT {index if label else index + 1}",
                    "result_equivalent": label,
                }
            )
    return {
        "run_id": f"run-{difficulty}",
        "dataset_suite": f"{difficulty}_sql_v1",
        "cases": cases,
        "aggregate": {},
    }


def test_sql_stratified_sampling_is_stable_and_records_shortfalls() -> None:
    runs = [
        _sql_run("easy", 8, 9),
        _sql_run("medium", 8, 9),
        _sql_run("hard", 5, 10),
    ]

    first = stratified_sample_sql_cases(runs, seed=7, reference_contexts=["DDL"])
    second = stratified_sample_sql_cases(runs, seed=7, reference_contexts=["DDL"])

    assert len(first.samples) == 30
    assert [item.case_id for item in first.samples] == [
        item.case_id for item in second.samples
    ]
    assert all(item.reference_contexts == ("DDL",) for item in first.samples)
    assert not any(first.shortfalls.values())

    shortage = stratified_sample_sql_cases(
        [_sql_run("easy", 2, 5), _sql_run("medium", 5, 5), _sql_run("hard", 3, 7)]
    )
    assert shortage.shortfalls["easy:positive"] == 3
    assert len(shortage.samples) == 27


def test_sql_scorer_and_summary_preserve_reason_latency_and_failure() -> None:
    labels = [True, True, False, False, False]
    samples = tuple(
        SqlJudgeSample(
            run_id="run",
            dataset_suite="suite",
            difficulty=("easy", "easy", "medium", "hard", "hard")[index],
            case_id=f"case-{index}",
            question="q",
            response_sql="SELECT 1",
            reference_sql="SELECT 1",
            reference_contexts=("DDL",),
            primary_label=label,
        )
        for index, label in enumerate(labels)
    )

    async def fake_scorer(sample):
        if sample.case_id == "case-4":
            raise TimeoutError("timeout")
        values = {"case-0": 1.0, "case-1": 0.0, "case-2": 1.0, "case-3": 0.0}
        return SqlMetricScore(values[sample.case_id], f"reason:{sample.case_id}")

    results = asyncio.run(score_sql_samples(samples, fake_scorer))
    summary = summarize_sql_judge_results(results)["overall"]

    assert len(results) == len(samples)
    assert results[0].reason == "reason:case-0"
    assert results[0].latency_ms >= 0
    assert results[-1].status == "error"
    assert results[-1].error_type == "TimeoutError"
    assert summary["confusion"] == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}
    assert summary["accuracy"] == 0.5
    assert summary["cohen_kappa"] == 0.0
    assert summary["error_count"] == 1
    assert summary["accuracy_wilson_95"]["low"] < 0.5


def test_full_sql_summary_keeps_deterministic_safety_performance_and_unknown_cost() -> None:
    easy = _sql_run("easy", 1, 1)
    easy.update(
        {
            "valid": False,
            "base_run_valid": True,
            "pipeline_evidence_eligible": False,
            "provider_request_ids_available": False,
        }
    )
    easy["cases"][0].update(
        {
            "result_evaluated": True,
            "syntax_valid": True,
            "exec_success": True,
            "structure_equivalent": True,
            "tables_correct": True,
            "table_scope_safe": True,
            "latency_ms": 10,
            "total_tokens": 100,
        }
    )
    easy["cases"][1].update(
        {
            "result_evaluated": True,
            "syntax_valid": False,
            "exec_success": False,
            "structure_equivalent": False,
            "tables_correct": False,
            "table_scope_safe": False,
            "latency_ms": 30,
            "total_tokens": 300,
        }
    )
    easy["aggregate"] = {
        "security_guard_bypass_count": 0,
        "infrastructure_errors": 1,
        "degraded_cases": 1,
        "latency_p50": 10,
        "latency_p95": 30,
        "tokens_mean": 200,
        "tokens_p95": 300,
        "cost_estimate_available": False,
        "estimated_cost_total": 0,
        "cost_currency": "USD",
    }

    summary = summarize_full_sql_runs([easy])
    overall = summary["overall"]

    assert overall["deterministic"]["result_equivalent"] == {
        "count": 1,
        "rate": 0.5,
    }
    assert overall["safety"] == {
        "security_guard_bypass_count": 0,
        "table_scope_unsafe_count": 1,
        "infrastructure_error_count": 1,
        "degraded_case_count": 1,
    }
    assert overall["latency_ms"]["p95"] == 30.0
    assert overall["tokens"]["total"] == 400
    assert overall["cost"]["available"] is False
    assert overall["cost"]["estimated_total"] is None
    assert overall["governance"]["all_base_runs_valid"] is True
    assert overall["governance"]["all_runs_valid"] is False


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_full_orchestration_uses_injected_scorers_without_composite(tmp_path: Path) -> None:
    cases, qrels, live_run = _retrieval_fixture()
    cases_path = tmp_path / "cases.csv"
    qrels_path = tmp_path / "qrels.csv"
    rag_path = tmp_path / "rag.json"
    sql_path = tmp_path / "sql.json"
    _write_csv(cases_path, cases)
    _write_csv(qrels_path, qrels)
    rag_path.write_text(json.dumps(live_run, ensure_ascii=False), encoding="utf-8")
    sql_path.write_text(
        json.dumps(_sql_run("easy", 1, 0), ensure_ascii=False), encoding="utf-8"
    )

    async def retrieval_scorer(sample):
        return IdMetricScores(
            deterministic_id_precision(sample.reference_ids, sample.retrieved_ids),
            deterministic_id_recall(sample.reference_ids, sample.retrieved_ids),
        )

    async def sql_scorer(_sample):
        return SqlMetricScore(1.0, "equivalent")

    report = asyncio.run(
        run_ragas_analysis(
            cases_path=cases_path,
            qrels_path=qrels_path,
            rag_live_run_path=rag_path,
            sql_run_paths=[sql_path],
            retrieval_scorer=retrieval_scorer,
            sql_scorer=sql_scorer,
            quotas={"easy": {True: 1, False: 0}},
        )
    )

    assert report["schema_version"] == "ragas-da-analysis-v1"
    assert "composite_score" not in report
    assert report["retrieval"]["summary"]["sample_count"] == 7
    assert report["sql"]["judge_summary"]["overall"]["computed_count"] == 1


def test_ddl_extractor_keeps_only_five_permanent_tables(tmp_path: Path) -> None:
    ddl_path = tmp_path / "schema.sql"
    ddl_path.write_text(
        "\n".join(
            [
                *(f"CREATE TABLE table_{index} (id INT);" for index in range(5)),
                "CREATE TEMPORARY TABLE scratch (id INT);",
                "INSERT INTO table_0 VALUES (1);",
                "CREATE TABLE table_6 (id INT);",
            ]
        ),
        encoding="utf-8",
    )

    contexts = extract_schema_ddl(ddl_path)

    assert len(contexts) == 5
    assert all("CREATE TABLE table_" in item for item in contexts)
    assert all("TEMPORARY" not in item and "INSERT" not in item for item in contexts)


def test_artifact_writer_serializes_json_and_per_sample_csv(tmp_path: Path) -> None:
    report = {
        "schema_version": "ragas-da-analysis-v1",
        "retrieval": {
            "results": [
                {
                    "case_id": "RAG-001",
                    "route": "table",
                    "reference_ids": ["table:a"],
                    "retrieved_ids": ["table:a", "table:b"],
                    "status": "computed",
                }
            ]
        },
        "sql": {
            "judge_results": [
                {
                    "case_id": "easy-001",
                    "status": "computed",
                    "primary_label": True,
                    "predicted_label": True,
                    "reason": "一致",
                }
            ]
        },
    }

    refs = write_analysis_artifacts(
        report,
        output_root=tmp_path,
        evaluation_id="ragas-test",
        run_id="run-1",
    )

    analysis_path = tmp_path / refs["analysis"].relative_path
    retrieval_path = tmp_path / refs["retrieval_csv"].relative_path
    sql_path = tmp_path / refs["sql_judge_csv"].relative_path
    assert json.loads(analysis_path.read_text(encoding="utf-8"))["schema_version"] == (
        "ragas-da-analysis-v1"
    )
    assert "RAG-001" in retrieval_path.read_text(encoding="utf-8-sig")
    assert "easy-001" in sql_path.read_text(encoding="utf-8-sig")


def test_sql_scorer_overrides_ragas_default_output_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: object) -> None:
            captured["client_kwargs"] = kwargs

    def fake_llm_factory(
        model: str, *, provider: str, client: object, **kwargs: object
    ) -> object:
        captured.update(
            {"model": model, "provider": provider, "client": client, **kwargs}
        )
        return object()

    openai_module = ModuleType("openai")
    openai_module.AsyncOpenAI = FakeAsyncOpenAI  # type: ignore[attr-defined]
    ragas_module = ModuleType("ragas")
    ragas_llms_module = ModuleType("ragas.llms")
    ragas_llms_base_module = ModuleType("ragas.llms.base")
    ragas_llms_base_module.llm_factory = fake_llm_factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", openai_module)
    monkeypatch.setitem(sys.modules, "ragas", ragas_module)
    monkeypatch.setitem(sys.modules, "ragas.llms", ragas_llms_module)
    monkeypatch.setitem(sys.modules, "ragas.llms.base", ragas_llms_base_module)
    monkeypatch.setattr(
        ragas_analysis, "ragas_runtime_metadata", lambda **_kwargs: {"compatible": True}
    )
    monkeypatch.setattr(
        ragas_analysis,
        "create_ragas_sql_semantic_scorer",
        lambda *, llm: llm,
    )

    scorer, _client = ragas_analysis.create_openai_compatible_sql_scorer(
        api_key="masked-test-key",
        base_url="https://judge.invalid",
        model="judge-model",
    )

    assert scorer is not None
    assert captured["temperature"] == 0.0
    assert captured["max_tokens"] == 4096
    assert "max_completion_tokens" not in captured
    assert captured["client_kwargs"] == {
        "api_key": "masked-test-key",
        "base_url": "https://judge.invalid",
        "timeout": 120.0,
    }

    with pytest.raises(ValueError, match="max_tokens"):
        ragas_analysis.create_openai_compatible_sql_scorer(
            api_key="masked-test-key",
            base_url="https://judge.invalid",
            model="judge-model",
            max_tokens=0,
        )

    with pytest.raises(ValueError, match="timeout_seconds"):
        ragas_analysis.create_openai_compatible_sql_scorer(
            api_key="masked-test-key",
            base_url="https://judge.invalid",
            model="judge-model",
            timeout_seconds=0,
        )
