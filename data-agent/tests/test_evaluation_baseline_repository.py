import unittest

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.domain.data_sources.context import stable_fingerprint
from app.domain.query.safety.semantic_violation import stable_hash as semantic_sql_hash
from evals.legacy.governance import REQUIRED_FINGERPRINT_FIELDS
from app.infrastructure.persistence.mysql.records.base import Base
from app.infrastructure.persistence.mysql.records.evaluation_baseline_mysql import EvaluationBaselineMySQL
from app.infrastructure.persistence.mysql.records.workspace_mysql import WorkspaceMySQL
from app.agent_system.agents.nl2sql.contracts.query_signals import stable_signal_hash
from evals.infrastructure.persistence.mysql.evaluation_baseline_repository import (
    EvaluationBaselineRepository,
)


def pipeline_aggregate_evidence(case_count: int = 1) -> dict:
    return {
        "query_signals": {
            "schema_version": "evaluation-query-signal-aggregate-v2",
            "active_profile_counts": {"jieba_tfidf_v1": case_count},
            "route_policy_counts": {"current_multi_query_v1": case_count},
            "dictionary_version_counts": {"none": case_count},
            "route_counts": {
                route: case_count
                for route in ("column", "metric", "training", "value")
            },
            "query_encode_mode_counts": {
                "column:query": case_count,
                "metric:query": case_count,
                "training:query": case_count,
                "value:not_applicable_es_lexical": case_count,
            },
            "actual_remote_signal_counts": {
                route: case_count
                for route in ("column", "metric", "training", "value")
            },
            "actual_consumer_signal_counts": {
                f"{route}:{consumer}": case_count
                for route in ("column", "metric", "training", "value")
                for consumer in ("local", "remote", "rerank")
            },
            "extractor_kind_counts": {"standalone_query": case_count},
            "consumer_kind_counts": {
                f"{route}:{consumer}:standalone_query": case_count
                for route in ("column", "metric", "training", "value")
                for consumer in ("local", "remote", "rerank")
            },
            "evidence_count": case_count,
            "evidence_complete_count": case_count,
            "route_evidence_count": case_count * 4,
            "dictionary_fallback_count": 0,
            "evidence_complete": True,
        },
        "planning_route": {
            "schema_version": "evaluation-planning-route-aggregate-v1",
            "active_profile_counts": {"keyword_length_v1": case_count},
            "configured_profile_counts": {"keyword_length_v1": case_count},
            "fallback_profile_counts": {"keyword_length_v1": case_count},
            "fallback_class_counts": {"none": case_count},
            "route_counts": {"direct_generate": case_count},
            "plan_calls": 0,
            "decide_node_calls": case_count,
            "decide_node_successes": case_count,
            "observed_plan_node_calls": 0,
            "plan_node_successes": 0,
            "physical_llm_calls": 0,
            "physical_llm_attempts": 0,
            "fallback_count": 0,
            "policy_fallback_count": 0,
            "unsafe_fallback_count": 0,
            "degraded_count": 0,
            "evidence_count": case_count,
            "evidence_complete_count": case_count,
            "decision_accounting_complete": True,
            "plan_call_accounting_complete": True,
            "physical_llm_call_accounting_complete": True,
            "evidence_complete": True,
        },
        "semantic_repair": {
            "schema_version": "evaluation-semantic-repair-aggregate-v1",
            "active_profile_counts": {"registry_same_rules_v1": case_count},
            "configured_profile_counts": {
                "registry_same_rules_v1": case_count,
            },
            "attempted_rule_counts": {
                "join.unique_metadata_edge": case_count,
                "projection.threshold_measure": case_count,
                "order.rank_limit_tie_breaker": case_count,
            },
            "applied_rule_counts": {},
            "changed_count": 0,
            "fallback_count": 0,
            "degraded_count": 0,
            "evidence_count": case_count,
            "evidence_complete_count": case_count,
            "invocation_count": case_count,
            "invocation_evidence_complete_count": case_count,
            "invocation_profile_counts": {
                "registry_same_rules_v1": case_count,
            },
            "invocation_phase_counts": {
                "primary_validate": case_count,
            },
            "selected_invocation_count": case_count,
            "invocation_fallback_count": 0,
            "ledger_complete_cases": case_count,
            "invocation_evidence_complete": True,
            "evidence_complete": True,
        },
    }


def pipeline_stage_evidence() -> dict:
    attempted = [
        "join.unique_metadata_edge",
        "projection.threshold_measure",
        "order.rank_limit_tie_breaker",
    ]
    sql_hash = semantic_sql_hash("SELECT 1")
    invariants = {
        "single_bounded_pass": True,
        "table_set_unchanged": True,
        "filter_literals_unchanged": True,
        "read_only_preserved": True,
    }

    source_fingerprint = stable_signal_hash({"fixture": "standalone-query"})
    extractor_signals = [{
        "signal_id": f"qs-{source_fingerprint[:12]}",
        "kind": "standalone_query",
        "rank": 0,
        "fingerprint": source_fingerprint,
    }]

    def query_route_evidence(route: str) -> dict:
        encode_mode = (
            "not_applicable_es_lexical" if route == "value" else "query"
        )

        def consumer_signals(consumer: str) -> list[dict]:
            return [{**extractor_signals[0], "consumer": consumer}]

        local_signals = consumer_signals("local")
        remote_signals = consumer_signals("remote")
        rerank_signals = consumer_signals("rerank")
        consumer_plan_fingerprint = stable_signal_hash({
            "contract": "query-signals-v1",
            "route_policy_version": "query-signal-routing-v2",
            "route_policy": "current_multi_query_v1",
            "route": route,
            "profile_fingerprint": "profile-fingerprint",
            "signal_fingerprint": "signal-fingerprint",
            "query_encode_mode": encode_mode,
            "local": local_signals,
            "remote": remote_signals,
            "rerank": rerank_signals,
        })
        return {
            "route": route,
            "extractor_profile": "jieba_tfidf_v1",
            "route_policy": "current_multi_query_v1",
            "route_policy_version": "query-signal-routing-v2",
            "dictionary_version": "none",
            "profile_fingerprint": "profile-fingerprint",
            "signal_fingerprint": "signal-fingerprint",
            "cache_namespace_fingerprint": f"cache-{route}",
            "query_encode_mode": encode_mode,
            "local_signal_count": 1,
            "remote_query_count": 1,
            "rerank_signal_count": 1,
            "actual_local_signal_count": 1,
            "actual_remote_signal_count": 1,
            "actual_rerank_signal_count": 1,
            "actual_local_signals": local_signals,
            "actual_remote_signals": remote_signals,
            "actual_rerank_signals": rerank_signals,
            "local_signal_fingerprint": stable_fingerprint(local_signals),
            "remote_signal_fingerprint": stable_fingerprint(remote_signals),
            "rerank_signal_fingerprint": stable_fingerprint(rerank_signals),
            "consumer_plan_fingerprint": consumer_plan_fingerprint,
            "consumer_kind_counts": {
                consumer: {"standalone_query": 1}
                for consumer in ("local", "remote", "rerank")
            },
            "evidence_complete": True,
            "evidence_failures": [],
            "raw_text_logged": False,
        }

    return {
        "query_signals": {
            "schema_version": "evaluation-query-signal-evidence-v2",
            "contract_version": "query-signals-v1",
            "configured_profile": "jieba_tfidf_v1",
            "active_profile": "jieba_tfidf_v1",
            "configured_route_policy": "current_multi_query_v1",
            "dictionary_version": "none",
            "profile_fingerprint": "profile-fingerprint",
            "signal_fingerprint": "signal-fingerprint",
            "extractor_signals": extractor_signals,
            "extractor_signal_identity_fingerprint": stable_fingerprint(
                extractor_signals
            ),
            "extractor_kind_counts": {"standalone_query": 1},
            "route_counts": {
                route: 1
                for route in ("column", "metric", "training", "value")
            },
            "routes": [
                query_route_evidence(route)
                for route in ("column", "metric", "training", "value")
            ],
            "dictionary_fallback_count": 0,
            "evidence_complete": True,
            "evidence_failures": [],
            "raw_text_logged": False,
        },
        "planning_route": {
            "schema_version": "evaluation-planning-route-evidence-v1",
            "contract_version": "planning-route-v1",
            "configured_profile": "keyword_length_v1",
            "runtime_profile": "keyword_length_v1",
            "active_profile": "keyword_length_v1",
            "runtime_override_active": False,
            "decision": "direct_generate",
            "route": "direct_generate",
            "plan_called": False,
            "physical_llm_call": False,
            "fallback_profile": "keyword_length_v1",
            "fallback_applied": False,
            "fallback_class": "none",
            "policy_fallback_eligible": False,
            "degraded": False,
            "repair_profile": "registry_same_rules_v1",
            "reason_codes": [],
            "evidence_complete": True,
            "evidence_failures": [],
            "raw_text_logged": False,
        },
        "semantic_repair": {
            "schema_version": "evaluation-semantic-repair-evidence-v1",
            "contract_version": "semantic-repair-result-v1",
            "registry_version": "semantic-repair-registry-v1",
            "rule_set_version": "semantic-repair-rules-v2",
            "configured_profile": "registry_same_rules_v1",
            "runtime_profile": "registry_same_rules_v1",
            "active_profile": "registry_same_rules_v1",
            "runtime_override_active": False,
            "fallback_profile": "inline_v1",
            "fallback_used": False,
            "degraded": False,
            "changed": False,
            "attempted_rule_codes": attempted,
            "applied_rules": [],
            "skipped_rules": [
                {"rule_code": code, "status": "skipped"}
                for code in attempted
            ],
            "invariants": invariants,
            "original_sql_hash": sql_hash,
            "repaired_sql_hash": sql_hash,
            "final_sql_hash": sql_hash,
            "ledger": [{
                "schema_version": "semantic-repair-ledger-v1",
                "phase": "primary_validate",
                "committed_at_phase": True,
                "selected_at_phase": True,
                "selected": True,
                "validation_valid": True,
                "contract_version": "semantic-repair-result-v1",
                "registry_version": "semantic-repair-registry-v1",
                "rule_set_version": "semantic-repair-rules-v2",
                "profile": "registry_same_rules_v1",
                "original_sql_hash": sql_hash,
                "repaired_sql_hash": sql_hash,
                "observed_sql_hash": sql_hash,
                "attempted_rule_codes": attempted,
                "applied_rules": [],
                "skipped_rules": [],
                "fallback_used": False,
                "fallback_reason": None,
                "invariants": invariants,
                "evidence_complete": True,
                "evidence_failures": [],
                "raw_sql_logged": False,
            }],
            "invocation_count": 1,
            "selected_invocation_count": 1,
            "ledger_evidence_complete": True,
            "ledger_truncated": False,
            "evidence_complete": True,
            "evidence_failures": [],
            "raw_sql_logged": False,
        },
    }


def eligible_run(run_id: str, dataset_suite: str = "easy_sql_v1") -> dict:
    fingerprint = {field: f"value-{field}" for field in REQUIRED_FINGERPRINT_FIELDS}
    fingerprint.update({
        "complete": True, "run_fingerprint": f"sha256:{run_id}",
        "evaluation_profile": {
            "schema": "evaluation-profile-v1",
            "name": "core-default",
            "analysis_enabled": False,
        },
        "sample_contract": {"total": 1, "complete": True, "all_reviewed": True},
        "pipeline_profiles": {
            "query_signals": {
                "profile": "jieba_tfidf_v1",
                "route_policy": "current_multi_query_v1",
                "workspace_dictionary_enabled": True,
            },
            "planning_route": {
                "active_profile": "keyword_length_v1",
                "fallback_profile": "keyword_length_v1",
                "repair_profile": "registry_same_rules_v1",
            },
            "semantic_repair": {
                "active_profile": "registry_same_rules_v1",
            },
        },
    })
    return {
        "run_id": run_id, "workspace_id": "workspace-a", "valid": True, "n": 1,
        "baseline_eligible": True,
        "dataset_suite": dataset_suite,
        "aggregate": {
            "infrastructure_errors": 0,
            "llm_attempt_schema": "llm-attempt-v2",
            "llm_attempt_accounting_complete": True,
            "failed_core_llm_calls": 0,
            "llm_attempts_timed_out": 0,
            "degradation_evidence_complete": True,
            **pipeline_aggregate_evidence(),
        },
        "fingerprint": fingerprint,
        "cases": [{"stage_evidence": {
            "retrieval": {}, "filtered_tables": [], "first_sql": "SELECT 1",
            "correction_sqls": [], "authorized_sql": "SELECT 1",
            "result_summary": {"masked": True, "values_persisted": False},
            "semantic_coverage": {
                "schema_version": "semantic-coverage-v1",
                "coverage_score": 1.0,
                "unresolved_slots": [],
            },
            "request_deadline": {
                "contract_version": "request-deadline-v1",
                "remaining_ms": 1000,
                "absolute_timestamps_persisted": False,
            },
            **pipeline_stage_evidence(),
        }}],
    }


class EvaluationBaselineRepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_async_engine(
            "sqlite+aiosqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool,
        )
        self.factory = async_sessionmaker(self.engine, expire_on_commit=False)
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with self.factory() as session:
            session.add(WorkspaceMySQL(
                id="workspace-a", name="Evaluation Workspace", owner_id="owner-a",
            ))
            await session.commit()

    async def asyncTearDown(self):
        await self.engine.dispose()

    async def test_approval_supersedes_previous_baseline_transactionally(self):
        first, second = eligible_run("run-1"), eligible_run("run-2")
        async with self.factory() as session:
            repo = EvaluationBaselineRepository(session)
            await repo.register_candidate(first, "run_run-1.json")
            await repo.register_candidate(second, "run_run-2.json")
            await repo.review(first, decision="approve", reviewer="alice")
            await repo.review(second, decision="approve", reviewer="bob")
            approved = await repo.approved("workspace-a", "easy_sql_v1")
            self.assertEqual(approved["run_id"], "run-2")
            self.assertEqual(approved["dataset_suite"], "easy_sql_v1")
            statuses = list((await session.execute(
                select(EvaluationBaselineMySQL.status).where(
                    EvaluationBaselineMySQL.workspace_id == "workspace-a"
                )
            )).scalars().all())
            self.assertEqual(statuses.count("approved"), 1)
            self.assertEqual(statuses.count("superseded"), 1)

    async def test_candidate_registration_is_idempotent(self):
        run = eligible_run("run-idempotent")
        async with self.factory() as session:
            repo = EvaluationBaselineRepository(session)
            first = await repo.register_candidate(run, "run-idempotent.json")
            second = await repo.register_candidate(run, "run-idempotent.json")
            self.assertEqual(first["run_id"], second["run_id"])
            rows = list((await session.execute(
                select(EvaluationBaselineMySQL).where(
                    EvaluationBaselineMySQL.workspace_id == "workspace-a",
                    EvaluationBaselineMySQL.run_id == "run-idempotent",
                )
            )).scalars())
            self.assertEqual(len(rows), 1)

    async def test_candidate_registration_rejects_identity_conflicts(self):
        run = eligible_run("run-conflict")
        async with self.factory() as session:
            repo = EvaluationBaselineRepository(session)
            await repo.register_candidate(run, "run-conflict.json")
            with self.assertRaisesRegex(ValueError, "artifact"):
                await repo.register_candidate(run, "different.json")
            with self.assertRaisesRegex(ValueError, "dataset_suite"):
                await repo.register_candidate(
                    {**run, "dataset_suite": "medium_sql_v1"},
                    "run-conflict.json",
                )

    async def test_three_suites_keep_independent_approved_baselines(self):
        runs = [
            eligible_run("easy-run", "easy_sql_v1"),
            eligible_run("medium-run", "medium_sql_v1"),
            eligible_run("hard-run", "hard_sql_v1"),
        ]
        async with self.factory() as session:
            repo = EvaluationBaselineRepository(session)
            for run in runs:
                await repo.register_candidate(run, f"{run['run_id']}.json")
                await repo.review(run, decision="approve", reviewer="alice")

            approved = await repo.approved_all("workspace-a")
            self.assertEqual(
                {item["dataset_suite"]: item["run_id"] for item in approved},
                {
                    "easy_sql_v1": "easy-run",
                    "medium_sql_v1": "medium-run",
                    "hard_sql_v1": "hard-run",
                },
            )
            self.assertTrue(all(item["status"] == "approved" for item in approved))

    async def test_approval_only_supersedes_within_the_same_suite(self):
        easy_first = eligible_run("easy-first", "easy_sql_v1")
        easy_second = eligible_run("easy-second", "easy_sql_v1")
        medium = eligible_run("medium-current", "medium_sql_v1")
        async with self.factory() as session:
            repo = EvaluationBaselineRepository(session)
            for run in (easy_first, medium, easy_second):
                await repo.register_candidate(run, f"{run['run_id']}.json")
                await repo.review(run, decision="approve", reviewer="alice")

            self.assertEqual(
                (await repo.approved("workspace-a", "easy_sql_v1"))["run_id"],
                "easy-second",
            )
            self.assertEqual(
                (await repo.approved("workspace-a", "medium_sql_v1"))["run_id"],
                "medium-current",
            )
            statuses = {
                row.run_id: row.status
                for row in (await session.execute(
                    select(EvaluationBaselineMySQL).where(
                        EvaluationBaselineMySQL.workspace_id == "workspace-a"
                    )
                )).scalars().all()
            }
            self.assertEqual(statuses["easy-first"], "superseded")
            self.assertEqual(statuses["easy-second"], "approved")
            self.assertEqual(statuses["medium-current"], "approved")

    async def test_nonrelease_artifact_cannot_enter_baseline_repository(self):
        run = {
            **eligible_run("model-boundary"),
            "baseline_eligible": False,
            "artifact_scope": "non_promotable_observational",
        }
        async with self.factory() as session:
            repo = EvaluationBaselineRepository(session)
            with self.assertRaisesRegex(ValueError, "non-release"):
                await repo.register_candidate(run, "model.json")
            with self.assertRaisesRegex(ValueError, "non-release"):
                await repo.review(
                    run, decision="approve", reviewer="alice",
                )
            rows = list((await session.execute(
                select(EvaluationBaselineMySQL)
            )).scalars())
            self.assertEqual(rows, [])

    async def test_candidate_and_review_require_explicit_true_eligibility(self):
        for index, value in enumerate((None, 1), start=1):
            run = eligible_run(f"eligibility-{index}")
            if value is None:
                run.pop("baseline_eligible")
            else:
                run["baseline_eligible"] = value
            async with self.factory() as session:
                repo = EvaluationBaselineRepository(session)
                with self.assertRaisesRegex(ValueError, "baseline_eligible"):
                    await repo.register_candidate(run, f"eligibility-{index}.json")
                with self.assertRaisesRegex(ValueError, "baseline_eligible"):
                    await repo.review(
                        run, decision="approve", reviewer="alice",
                    )
                rows = list((await session.execute(
                    select(EvaluationBaselineMySQL).where(
                        EvaluationBaselineMySQL.run_id
                        == f"eligibility-{index}"
                    )
                )).scalars())
                self.assertEqual(rows, [])

    async def test_forged_eligibility_cannot_hide_llm_release_failures(self):
        for index, raw_field in enumerate((
            "failed_core_llm_calls", "llm_attempts_timed_out",
        ), start=1):
            run = eligible_run(f"llm-release-failure-{index}")
            run["baseline_eligible"] = True
            run["aggregate"][raw_field] = 1
            async with self.factory() as session:
                repo = EvaluationBaselineRepository(session)
                with self.assertRaisesRegex(ValueError, "candidate 准入"):
                    await repo.register_candidate(
                        run, f"llm-release-failure-{index}.json",
                    )
                rows = list((await session.execute(
                    select(EvaluationBaselineMySQL).where(
                        EvaluationBaselineMySQL.run_id == run["run_id"]
                    )
                )).scalars())
                self.assertEqual(rows, [])

    async def test_legacy_run_without_pipeline_evidence_cannot_be_approved(self):
        run = eligible_run("legacy-without-pipeline-evidence")
        run["aggregate"].pop("planning_route")
        run["aggregate"].pop("semantic_repair")
        run["cases"][0]["stage_evidence"].pop("planning_route")
        run["cases"][0]["stage_evidence"].pop("semantic_repair")
        async with self.factory() as session:
            repo = EvaluationBaselineRepository(session)
            with self.assertRaisesRegex(ValueError, "candidate 准入"):
                await repo.register_candidate(run, "legacy.json")
            with self.assertRaisesRegex(ValueError, "baseline 准入"):
                await repo.review(
                    run, decision="approve", reviewer="alice",
                )
            rows = list((await session.execute(
                select(EvaluationBaselineMySQL)
            )).scalars())
            self.assertEqual(rows, [])

    async def test_commit_expiration_is_refreshed_before_serialization(self):
        """MySQL server defaults must not trigger MissingGreenlet in view()."""
        expiring_factory = async_sessionmaker(self.engine, expire_on_commit=True)
        run = eligible_run("run-expiring")
        async with expiring_factory() as session:
            repo = EvaluationBaselineRepository(session)
            candidate = await repo.register_candidate(run, "run_expiring.json")
            self.assertEqual(candidate["run_id"], "run-expiring")
            self.assertIsNotNone(candidate["created_at"])
            reviewed = await repo.review(
                run, decision="reject", reviewer="alice", note="regression",
            )
            self.assertEqual(reviewed["status"], "rejected")
            self.assertEqual(reviewed["reviewed_by"], "alice")

if __name__ == "__main__":
    unittest.main()
