"""Re-evaluate stored generated SQL after a deterministic metric-contract change.

This command never calls the LLM.  It executes the generated/reference SQL saved in
one or more existing runs, preserves the previous verdict for audit, and updates only
derived result-equivalence aggregates.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
from pathlib import Path

from app.infrastructure.clients.mysql_client_manager import dw_mysql_client_manager
from evals.legacy.dataset_suite import SUITE_SPECS
from evals.legacy.governance import is_explicit_nonrelease
from evals.legacy.metrics import (
    RESULT_EQUIVALENCE_CONTRACT_VERSION,
    question_requires_result_order,
    result_equivalence_evidence,
)
from evals.legacy.run_eval import (
    aggregate,
    aggregate_by_category,
    load_cases,
    render_report,
)
from app.infrastructure.persistence.mysql.repositories.dw_mysql_repository import DWMysqlRepository


def _safe_db_error_code(error: Exception) -> int | None:
    original = getattr(error, "orig", None)
    for candidate in (original, error):
        args = getattr(candidate, "args", ())
        if args and isinstance(args[0], int):
            return args[0]
    return None


async def recheck(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("valid"):
        raise ValueError("只能复核 valid=true 的真实评测产物")
    if is_explicit_nonrelease(payload):
        raise ValueError("non-release/sealed 评测制品不可原地复核或覆盖发布报告")
    dataset_suite = str(payload.get("dataset_suite") or "").strip()
    if dataset_suite not in SUITE_SPECS:
        raise ValueError("运行制品缺少受支持的 dataset_suite，不能按旧默认值复核")
    contracts = {
        case["id"]: case
        for case in load_cases(suite=dataset_suite)
    }

    dw_mysql_client_manager.init()
    try:
        async with dw_mysql_client_manager.session_factory() as session:
            repo = DWMysqlRepository(session)
            for item in payload["cases"]:
                case_id = str(item.get("id") or "")
                if case_id not in contracts:
                    raise ValueError(
                        f"运行制品用例 {case_id!r} 不属于数据集 {dataset_suite!r}"
                    )
                contract = contracts[case_id]
                audit_suffix = f"contract_v{RESULT_EQUIVALENCE_CONTRACT_VERSION}"
                item.setdefault(
                    f"result_equivalent_before_{audit_suffix}",
                    item["result_equivalent"],
                )
                item.setdefault(
                    f"result_ordered_before_{audit_suffix}",
                    item.get("result_ordered"),
                )
                question = str(contract.get("question") or item.get("question") or "")
                item["result_ordered_declared"] = bool(
                    contract.get("result_ordered", False)
                )
                item["result_ordered"] = question_requires_result_order(question)
                # Keep both observations complete and transaction-isolated, in
                # parity with the live evaluator.  Product LIMIT/cache behaviour
                # must not become part of the recheck denominator.
                await session.rollback()
                try:
                    expected_rows = await repo.execute_sql_for_evaluation(
                        item["expected_sql"]
                    )
                except Exception as exc:
                    raise ValueError(
                        f"Reference SQL 复核失败：{case_id} ({type(exc).__name__})"
                    ) from exc
                await session.rollback()
                try:
                    generated_rows = await repo.execute_sql_for_evaluation(
                        item["gen_sql"]
                    )
                except Exception as exc:
                    item["result_equivalence_evidence"] = {
                        "contract_version": RESULT_EQUIVALENCE_CONTRACT_VERSION,
                        "equivalent": False,
                        "comparison_mode": "generated_execution_error",
                        "order_sensitive": item["result_ordered"],
                        "minimum_sufficient_projection": {
                            "attempted": False,
                            "eligible": False,
                            "reason": "generated_execution_error",
                        },
                    }
                    item["result_equivalent"] = False
                    item["recheck_execution"] = {
                        "status": "generated_execution_error",
                        "error_type": type(exc).__name__,
                        "db_error_code": _safe_db_error_code(exc),
                    }
                    continue
                comparison = result_equivalence_evidence(
                    generated_rows,
                    expected_rows,
                    item["expected_sql"],
                    question=question,
                    generated_sql=item["gen_sql"],
                )
                item["result_equivalence_evidence"] = comparison
                item["result_equivalent"] = bool(comparison["equivalent"])
                item["recheck_execution"] = {"status": "compared"}
    finally:
        await dw_mysql_client_manager.close()

    previous = payload.get("aggregate", {})
    payload["recheck_history"] = [
        *(payload.get("recheck_history") or []),
        {
            "at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "reason": (
                "按用户问题的显式顺序语义、完整结果分母和有界数值容差复核；"
                "Reference ORDER BY 与旧 result_ordered 标志不再推断业务顺序；"
                "双方投影仅按唯一语义映射比较，题面要求列不得省略；"
                "仅对已解析档位序号和可加总指标的非输出粒度做归一；"
                "文本、字段与整数计数仍严格比较"
            ),
            "previous_result_equivalent": previous.get("result_equivalent"),
            "contract_version": RESULT_EQUIVALENCE_CONTRACT_VERSION,
        },
    ]
    payload["evaluation_contract_version"] = RESULT_EQUIVALENCE_CONTRACT_VERSION
    payload["aggregate"] = aggregate(payload["cases"])
    payload["by_category"] = aggregate_by_category(payload["cases"])
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    report = render_report(payload["cases"], payload["aggregate"], payload["by_category"])
    report_dir = path.parents[1] / "reports" / "nl2sql"
    report_dir.mkdir(parents=True, exist_ok=True)
    suite_spec = SUITE_SPECS.get(dataset_suite) or {}
    artifact_ids = {str(item.get("id") or "") for item in payload.get("cases") or []}
    contract_ids = set(contracts)
    release_full_suite = bool(
        suite_spec.get("promotion_status") == "approved"
        and suite_spec.get("baseline_eligible", True)
        and artifact_ids == contract_ids
    )
    if release_full_suite:
        (report_dir / f"report_{dataset_suite}_latest.md").write_text(
            report, encoding="utf-8",
        )
    contract_version = RESULT_EQUIVALENCE_CONTRACT_VERSION
    (report_dir / f"report_{payload['run_id']}_contract_v{contract_version}.md").write_text(
        report,
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="不调用 LLM，按最新确定性指标契约复核 run")
    parser.add_argument(
        "run",
        type=Path,
        nargs="+",
        help="一个或多个 var/evals/legacy/runs/run_<id>.json",
    )
    args = parser.parse_args()

    async def recheck_all() -> list[dict]:
        return [await recheck(path.resolve()) for path in args.run]

    for payload in asyncio.run(recheck_all()):
        aggregate_data = payload["aggregate"]
        print(
            f"复核完成：{payload['run_id']}，结果一致 "
            f"{aggregate_data['result_equivalent']}/{aggregate_data['n']}，"
            f"P95 {aggregate_data['latency_p95']} ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
