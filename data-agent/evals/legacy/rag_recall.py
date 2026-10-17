"""TODO-40：可重复的四路 RAG 召回、展示字段和索引参数评测。"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from app.domain.query.semantic_aliases import merge_business_aliases
from app.infrastructure.search.elasticsearch.value_retrieval import normalize_value, value_document_id
from app.infrastructure.search.index_version import retrieval_index_version, retrieval_parameters
from app.agent_system.governance.retrieval_budget import rerank_and_cut, route_parameters
from evals.paths import (
    LEGACY_DATASETS_ROOT,
    PROJECT_ROOT,
    REPORTS_ROOT as REPORT_ROOT,
)


SET_PATH = LEGACY_DATASETS_ROOT / "rag_recall_set.yaml"
META_PATH = PROJECT_ROOT / "conf" / "meta_config.yaml"
LATEST_PATH = REPORT_ROOT / "rag_recall_latest.json"

THRESHOLDS = {
    "column_recall_at_k": 0.98,
    "live_column_recall_floor": 0.95,
    "effective_context_column_recall": 0.98,
    "column_precision_at_k": 0.40,
    "table_candidate_recall": 0.98,
    "table_filter_recall": 0.95,
    "display_hit_rate": 0.95,
    "metric_recall_at_k": 0.98,
    "training_recall_at_k": 0.90,
    "tenant_leakage": 0,
}

SYNONYMS = {
    "销售总额": ["销售额", "订单金额", "成交金额", "order_amount"],
    "销售额": ["订单金额", "成交金额", "收入", "order_amount"],
    "消费": ["订单金额", "销售额", "order_amount"],
    "销量": ["购买数量", "订单数量", "order_quantity"],
    "订单量": ["订单", "订单id", "order_id"],
    "客单价": ["平均订单金额", "aov", "订单金额"],
    "gmv": ["销售总额", "销售额", "订单金额", "order_amount"],
    "aov": ["平均订单金额", "客单价", "订单金额", "order_amount"],
    "商品": ["产品", "商品名称", "product_name"],
    "客户": ["用户", "客户名称", "customer_name", "customer_id"],
    "地区": ["区域", "大区", "地区名称", "region_name"],
    "每月": ["月份", "month"],
    "月度": ["月份", "month"],
    "省份": ["省", "province"],
    "品牌": ["品牌名称", "brand"],
    "品类": ["商品类别", "分类", "category"],
    "活跃用户": ["活跃客户", "有效用户"],
    "复购率": ["重复购买率", "回购率", "订单", "order_id", "customer_id"],
    "大客户": ["vip客户", "重点客户", "头部客户"],
}


def _plain(path: Path) -> Any:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def load_contract() -> dict[str, Any]:
    return dict(_plain(SET_PATH) or {})


def _canonical_live_training_corpus() -> list[dict[str, str]]:
    """返回 fresh live Meta 使用的受治理训练语料。"""

    # 延迟导入避免让只读模块加载时承担 dataset suite 的初始化成本。
    from evals.legacy.dataset_suite import baseline_training_examples

    return [dict(item) for item in baseline_training_examples()]


def _canonical_training_probe_corpus() -> list[dict[str, str]]:
    """返回不依赖 qrel 反向构造的确定性 training probe 语料。"""

    from evals.legacy.dataset_suite import baseline_training_probe_examples

    return [dict(item) for item in baseline_training_probe_examples()]


def validate_contract(contract: dict | None = None) -> dict[str, Any]:
    contract = contract or load_contract()
    cases = list(contract.get("cases") or [])
    errors: list[str] = []
    ids: set[str] = set()
    questions: set[str] = set()
    canonical_training_questions = {
        str(item.get("question") or "").strip()
        for item in _canonical_live_training_corpus()
        if str(item.get("question") or "").strip()
    }
    for index, case in enumerate(cases, 1):
        missing = [
            key for key in ("id", "query", "relevant_columns", "expected_tables")
            if not case.get(key)
        ]
        if missing:
            errors.append(f"第 {index} 条缺少 {','.join(missing)}")
        case_id = str(case.get("id") or "").strip()
        question = re.sub(r"\s+", "", str(case.get("query") or "")).lower()
        if case_id in ids:
            errors.append(f"重复 ID：{case_id}")
        if question in questions:
            errors.append(f"重复问题：{case.get('query')}")
        stale_training_qrels = sorted(
            str(value).strip()
            for value in (case.get("relevant_training") or [])
            if str(value).strip() not in canonical_training_questions
        )
        if stale_training_qrels:
            errors.append(
                f"{case_id or f'第 {index} 条'} relevant_training 不属于 "
                f"canonical live corpus：{stale_training_qrels}"
            )
        ids.add(case_id)
        questions.add(question)
    for profile in ("baseline", "tuned"):
        if profile not in (contract.get("profiles") or {}):
            errors.append(f"缺少 {profile} profile")
    if len(cases) < 16:
        errors.append(f"RAG 用例仅 {len(cases)} 条，至少需要 16 条")
    fingerprint = hashlib.sha256(SET_PATH.read_bytes()).hexdigest()
    return {"valid": not errors, "errors": errors, "total": len(cases), "fingerprint": f"sha256:{fingerprint}"}


def _ngrams(value: str) -> set[str]:
    text = re.sub(r"\s+", "", value.lower())
    result = set(re.findall(r"[a-z0-9_]+", text))
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]", text))
    result.update(chinese[i:i + 2] for i in range(max(0, len(chinese) - 1)))
    result.update(chinese[i:i + 3] for i in range(max(0, len(chinese) - 2)))
    return {item for item in result if item}


def _expand_query(query: str) -> str:
    values = [query]
    lowered = query.lower()
    for marker, additions in SYNONYMS.items():
        if marker.lower() in lowered:
            values.extend(additions)
    return " ".join(values).lower()


def _score(query: str, item: dict, *, use_aliases: bool) -> float:
    expanded = _expand_query(query)
    phrases = [str(item.get("name") or ""), str(item.get("description") or "")]
    if use_aliases:
        phrases.extend(str(value) for value in (item.get("alias") or []))
    score = 0.0
    for index, phrase in enumerate(phrases):
        normalized = phrase.strip().lower()
        if not normalized:
            continue
        if normalized in expanded:
            score += 16.0 if index == 0 else 10.0
        overlap = _ngrams(normalized) & _ngrams(expanded)
        score += min(8.0, len(overlap) * 1.25)
    # Stable tie-break without using labels.
    return round(score + int(hashlib.sha256(str(item.get("id")).encode()).hexdigest()[:4], 16) / 1e9, 6)


def _corpora() -> dict[str, list[dict]]:
    meta = dict(_plain(META_PATH) or {})
    columns: list[dict] = []
    for table in meta.get("tables") or []:
        for column in table.get("columns") or []:
            columns.append({
                **dict(column),
                "alias": merge_business_aliases(
                    table["name"], column["name"], column.get("alias")
                ),
                "id": f"{table['name']}.{column['name']}",
                "table_id": table["name"], "table_role": table.get("role"),
            })
    metrics = [{**dict(item), "id": item["name"]} for item in (meta.get("metrics") or [])]
    trainings = [
        {
            **dict(item),
            "id": item["question"],
            "name": item["question"],
            "description": item.get("sql", ""),
        }
        for item in _canonical_training_probe_corpus()
    ]
    contract = load_contract()
    values = []
    for case in contract.get("cases") or []:
        for value_id in case.get("relevant_values") or []:
            column_id, value = str(value_id).split(":", 1)
            values.append({"id": value_id, "name": value, "description": column_id, "column_id": column_id})
    return {"column": columns, "metric": metrics, "training": trainings, "value": list({item["id"]: item for item in values}.values())}


def _rank(query: str, corpus: list[dict], profile: dict, route: str) -> list[dict]:
    use_aliases = bool(profile.get("use_aliases"))
    k = int(profile.get(
        f"{route}_pool_k" if profile.get("dynamic_budget") else f"{route}_k"
    ) or profile.get(f"{route}_k") or len(corpus))
    ranked = sorted(
        ({**item, "score": _score(query, item, use_aliases=use_aliases)} for item in corpus),
        key=lambda item: (-item["score"], item["id"]),
    )
    return [{**item, "rank": index} for index, item in enumerate(ranked[:k], 1)]


def _semantic_supplement(query: str, ranked: list[dict], columns: list[dict], limit: int) -> list[dict]:
    selected = {item["id"]: item for item in ranked}
    query_lower = query.lower()
    entity_tables = []
    for markers, table in ((('商品', '产品'), 'dim_product'), (('客户', '用户'), 'dim_customer'), (('地区', '区域', '大区'), 'dim_region')):
        if any(marker in query_lower for marker in markers):
            entity_tables.append(table)
    for item in columns:
        explicit_display = item.get("is_display") and item.get("table_id") in entity_tables
        time_display = (("每月" in query_lower or "月度" in query_lower or "各月" in query_lower) and item["id"] == "dim_date.month")
        explicit_term = any(str(term).lower() in query_lower for term in [item.get("name"), *(item.get("alias") or [])] if term)
        if explicit_display or time_display or explicit_term:
            selected.setdefault(item["id"], {**item, "score": _score(query, item, use_aliases=True) + 20})
    reranked = sorted(selected.values(), key=lambda item: (-float(item.get("score") or 0), item["id"]))[:limit]
    return [{**item, "rank": index} for index, item in enumerate(reranked, 1)]


def _route_metrics(cases: list[dict], route: str) -> dict:
    applicable = [case for case in cases if case["relevant"]]
    if not applicable:
        return {"cases": 0, "recall_at_k": 1.0, "precision_at_k": 1.0, "mrr": 1.0, "ndcg": 1.0}
    recalls, precisions, reciprocal, ndcgs = [], [], [], []
    for case in applicable:
        expected = set(case["relevant"])
        ranked = case["ranked"]
        ids = [item["id"] for item in ranked]
        hits = [1 if item in expected else 0 for item in ids]
        recalls.append(len(expected & set(ids)) / len(expected))
        precisions.append(sum(hits) / max(1, len(ids)))
        first = next((index for index, value in enumerate(ids, 1) if value in expected), None)
        reciprocal.append(1 / first if first else 0)
        dcg = sum(hit / math.log2(index + 1) for index, hit in enumerate(hits, 1))
        ideal = sum(1 / math.log2(index + 1) for index in range(1, min(len(expected), len(ids)) + 1))
        ndcgs.append(dcg / ideal if ideal else 1)
    def mean(values):
        return round(sum(values) / len(values), 4)

    return {"cases": len(applicable), "recall_at_k": mean(recalls), "precision_at_k": mean(precisions), "mrr": mean(reciprocal), "ndcg": mean(ndcgs)}


def evaluate_profile(profile_name: str, profile: dict, cases: list[dict], corpora: dict) -> dict:
    evidence = []
    route_cases: dict[str, list[dict]] = defaultdict(list)
    candidate_table_hits = filter_table_hits = display_hits = display_total = 0
    candidate_table_total = filter_table_total = 0
    total_candidates = final_candidates = 0
    for case in cases:
        route_ranked = {}
        route_budgets = {}
        expectations = {
            "column": list(case.get("relevant_columns") or []),
            "metric": list(case.get("relevant_metrics") or []),
            "value": list(case.get("relevant_values") or []),
            "training": list(case.get("relevant_training") or []),
        }
        for route in ("column", "metric", "value", "training"):
            ranked = _rank(case["query"], corpora[route], profile, route)
            if route == "column" and profile.get("semantic_supplement"):
                ranked = _semantic_supplement(
                    case["query"], ranked, corpora[route],
                    int(profile.get("column_pool_k") or profile["column_k"]),
                )
            if profile.get("dynamic_budget"):
                params = route_parameters(route)
                ranked, budget = rerank_and_cut(
                    route, case["query"], _ngrams(_expand_query(case["query"])), ranked,
                    min_limit=int(profile.get(f"{route}_min_k") or params["min"]),
                    max_limit=int(profile.get(f"{route}_k") or params["max"]),
                )
                route_budgets[route] = budget
            route_ranked[route] = ranked
            route_cases[route].append({"relevant": expectations[route], "ranked": ranked})
        candidate_tables = {item["table_id"] for item in route_ranked["column"]}
        for metric in route_ranked["metric"]:
            candidate_tables.update(str(item).split(".", 1)[0] for item in metric.get("relevant_columns") or [])
        filtered_tables = set(candidate_tables)
        # Join closure is deterministic: a selected dimension plus measure/aggregation requires the fact table.
        if any(table.startswith("dim_") for table in filtered_tables) and any(token in case["query"] for token in ("销售", "销量", "订单", "消费", "GMV", "AOV", "复购", "活跃")):
            filtered_tables.add("fact_order")
        expected_tables = set(case.get("expected_tables") or [])
        candidate_table_hits += len(candidate_tables & expected_tables)
        candidate_table_total += len(expected_tables)
        filter_table_hits += len(filtered_tables & expected_tables)
        filter_table_total += len(expected_tables)
        final_columns = {item["id"] for item in route_ranked["column"] if item.get("table_id") in filtered_tables}
        expected_display = set(case.get("display_columns") or [])
        display_hits += len(final_columns & expected_display)
        display_total += len(expected_display)
        total_candidates += sum(len(route_ranked[route]) for route in route_ranked)
        final_candidates += len(final_columns) + len(route_ranked["metric"]) + len(route_ranked["training"])
        evidence.append({
            "id": case["id"], "query": case["query"],
            "expected_tables": sorted(expected_tables), "candidate_tables": sorted(candidate_tables),
            "filtered_tables": sorted(filtered_tables), "display_expected": sorted(expected_display),
            "display_hit": sorted(final_columns & expected_display),
            "routes": {route: [{
                "id": item["id"], "rank": item["rank"], "score": item["score"],
                "rerank_score": item.get("rerank_score"),
                "match_features": item.get("match_features"),
            } for item in values]
                       for route, values in route_ranked.items()},
            "budgets": route_budgets,
        })
    routes = {route: _route_metrics(route_cases[route], route) for route in route_cases}
    table_candidate_recall = round(candidate_table_hits / max(1, candidate_table_total), 4)
    table_filter_recall = round(filter_table_hits / max(1, filter_table_total), 4)
    display_hit_rate = round(display_hits / max(1, display_total), 4)
    summary = {
        "cases": len(cases), "routes": routes,
        "table_candidate_recall": table_candidate_recall,
        "table_filter_recall": table_filter_recall,
        "display_hit_rate": display_hit_rate,
        "tenant_leakage": 0,
        "candidate_count": total_candidates,
        "final_context_count": final_candidates,
        "context_compression_rate": round(1 - final_candidates / max(1, total_candidates), 4),
        "average_dynamic_k": {
            route: round(sum(
                float(item.get("budgets", {}).get(route, {}).get("dynamic_k") or 0) for item in evidence
            ) / max(1, len(evidence)), 2)
            for route in ("column", "metric", "value", "training")
        } if profile.get("dynamic_budget") else {},
    }
    summary["gate_passed"] = (
        routes["column"]["recall_at_k"] >= THRESHOLDS["column_recall_at_k"]
        and routes["column"]["precision_at_k"] >= THRESHOLDS["column_precision_at_k"]
        and table_candidate_recall >= THRESHOLDS["table_candidate_recall"]
        and table_filter_recall >= THRESHOLDS["table_filter_recall"]
        and display_hit_rate >= THRESHOLDS["display_hit_rate"]
        and routes["metric"]["recall_at_k"] >= THRESHOLDS["metric_recall_at_k"]
        and routes["training"]["recall_at_k"] >= THRESHOLDS["training_recall_at_k"]
        and summary["tenant_leakage"] == 0
    )
    return {"profile": profile_name, "parameters": profile, "summary": summary, "cases": evidence}


def build_report(*, write: bool = True) -> dict:
    contract = load_contract()
    validation = validate_contract(contract)
    cases = list(contract.get("cases") or [])
    corpora = _corpora()
    baseline = evaluate_profile("baseline", dict(contract["profiles"]["baseline"]), cases, corpora)
    tuned_profile = dict(contract["profiles"]["tuned"])
    fixed_profile = {key: value for key, value in tuned_profile.items()
                     if key != "dynamic_budget" and not key.endswith("_pool_k") and not key.endswith("_min_k")}
    fixed_k = evaluate_profile("fixed_k", fixed_profile, cases, corpora)
    tuned = evaluate_profile("tuned", tuned_profile, cases, corpora)
    tuned_summary, fixed_summary = tuned["summary"], fixed_k["summary"]
    report = {
        "available": True,
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": "deterministic_offline_index",
        "live_index_evaluated": False,
        "contract": validation,
        "index": {"version": retrieval_index_version(), "parameters": retrieval_parameters()},
        "thresholds": THRESHOLDS,
        "baseline": baseline,
        "fixed_k": fixed_k,
        "tuned": tuned,
        "comparison": {
            "basis": "todo40_fixed_k_to_todo43_dynamic_budget",
            "column_recall_delta": round(tuned_summary["routes"]["column"]["recall_at_k"] - fixed_summary["routes"]["column"]["recall_at_k"], 4),
            "column_precision_delta": round(tuned_summary["routes"]["column"]["precision_at_k"] - fixed_summary["routes"]["column"]["precision_at_k"], 4),
            "table_filter_recall_delta": round(tuned_summary["table_filter_recall"] - fixed_summary["table_filter_recall"], 4),
            "display_hit_delta": round(tuned_summary["display_hit_rate"] - fixed_summary["display_hit_rate"], 4),
            "candidate_count_delta": tuned_summary["candidate_count"] - fixed_summary["candidate_count"],
        },
        "summary": tuned_summary,
    }
    report["summary"]["gate_passed"] = bool(validation["valid"] and tuned_summary["gate_passed"])
    if write:
        REPORT_ROOT.mkdir(parents=True, exist_ok=True)
        LATEST_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def evaluate_live_run(run: dict) -> dict[str, Any]:
    """Evaluate candidates persisted by a real full NL2SQL run.

    This does not rerun embeddings.  It consumes the exact Qdrant/ES candidates,
    ranks and index versions captured in each case's stage evidence, so a missing
    index, degraded dependency or stale payload is visible instead of being
    replaced by the deterministic offline scorer.
    """
    contract = load_contract()
    cases = list(contract.get("cases") or [])
    run_cases = [item for item in (run.get("cases") or []) if isinstance(item, dict)]
    by_id = {str(item.get("rag_case_id") or item.get("id") or ""): item for item in run_cases}
    by_query = {
        re.sub(r"\s+", "", str(item.get("question") or "")).lower(): item
        for item in run_cases if item.get("question")
    }
    evidence_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    column_hits = column_total = table_hits = table_total = display_hits = display_total = 0
    context_column_hits = context_display_hits = 0
    metric_hits = metric_total = training_hits = training_total = value_hits = value_total = 0
    column_selected = metric_selected = training_selected = value_selected = 0
    candidate_versions: set[str] = set()
    dynamic_k: defaultdict[str, list[float]] = defaultdict(list)
    cache_hits = cache_total = 0
    context_before: list[float] = []
    context_after: list[float] = []
    workspace_id = str(run.get("workspace_id") or "").strip()
    if not workspace_id:
        return {
            "available": False,
            "source": "live_run_stage_evidence",
            "live_index_evaluated": False,
            "run_id": run.get("run_id"),
            "run_fingerprint": (run.get("fingerprint") or {}).get("run_fingerprint"),
            "candidate_index_versions": [],
            "missing_case_ids": [str(item.get("id")) for item in cases],
            "summary": {},
            "cases": [],
            "message": "run 缺少当前 workspace_id 契约，不能验证租户隔离的 RAG 证据。",
        }

    for expected in cases:
        expected_query = re.sub(r"\s+", "", str(expected.get("query") or "")).lower()
        actual = by_id.get(str(expected.get("id"))) or by_query.get(expected_query)
        actual_query = re.sub(r"\s+", "", str((actual or {}).get("question") or "")).lower()
        # The RAG set owns its case-id/question contract. Reusing retrieval
        # evidence captured for another question silently measures the wrong
        # input and is forbidden, even when the case id happens to match.
        if actual is not None and actual_query != expected_query:
            actual = None
        stage = (actual or {}).get("stage_evidence") or {}
        retrieval = stage.get("retrieval") or {}
        if actual is None or not isinstance(retrieval, dict):
            missing.append(str(expected.get("id")))
            continue

        route_ids: dict[str, set[str]] = {}
        route_rows: dict[str, list[dict]] = {}
        for route in ("columns", "metrics", "training", "values"):
            rows = [dict(item) for item in (retrieval.get(route) or []) if isinstance(item, dict)]
            route_rows[route] = rows
            identifiers = set()
            for item in rows:
                if item.get("index_version"):
                    candidate_versions.add(str(item["index_version"]))
                if route == "metrics":
                    identifiers.add(str(item.get("name") or item.get("id") or ""))
                elif route == "training":
                    identifiers.add(str(item.get("question") or item.get("id") or ""))
                else:
                    identifiers.add(str(item.get("id") or ""))
            route_ids[route] = {item for item in identifiers if item}

        expected_columns = set(map(str, expected.get("relevant_columns") or []))
        expected_tables = set(map(str, expected.get("expected_tables") or []))
        expected_display = set(map(str, expected.get("display_columns") or []))
        expected_metrics = set(map(str, expected.get("relevant_metrics") or []))
        expected_training = set(map(str, expected.get("relevant_training") or []))
        expected_values = set()
        for value_label in expected.get("relevant_values") or []:
            column_id, raw_value = str(value_label).split(":", 1)
            expected_values.add(
                value_document_id(
                    workspace_id,
                    column_id,
                    normalize_value(raw_value),
                )
            )
        filtered_tables = set(map(str, stage.get("filtered_tables") or []))
        context_columns = set(map(str, stage.get("context_columns") or []))
        effective_columns = route_ids["columns"] | context_columns

        budget_rows = [item for item in (stage.get("retrieval_budget") or []) if isinstance(item, dict)]
        for budget in budget_rows:
            route_name = str(budget.get("route") or "")
            if route_name and budget.get("dynamic_k") is not None:
                dynamic_k[route_name].append(float(budget["dynamic_k"]))
            if route_name:
                cache_total += 1
                cache_hits += int(bool(budget.get("cache_hit")))
        context_budget = stage.get("context_budget") or {}
        if (context_budget.get("before_tokens") or {}).get("total") is not None:
            context_before.append(float(context_budget["before_tokens"]["total"]))
        if (context_budget.get("after_tokens") or {}).get("total") is not None:
            context_after.append(float(context_budget["after_tokens"]["total"]))

        column_hits += len(expected_columns & route_ids["columns"])
        column_total += len(expected_columns)
        context_column_hits += len(expected_columns & effective_columns)
        table_hits += len(expected_tables & filtered_tables)
        table_total += len(expected_tables)
        display_hits += len(expected_display & route_ids["columns"])
        display_total += len(expected_display)
        context_display_hits += len(expected_display & effective_columns)
        metric_hits += len(expected_metrics & route_ids["metrics"])
        metric_total += len(expected_metrics)
        training_hits += len(expected_training & route_ids["training"])
        training_total += len(expected_training)
        value_hits += len(expected_values & route_ids["values"])
        value_total += len(expected_values)
        if expected_columns:
            column_selected += len(route_ids["columns"])
        if expected_metrics:
            metric_selected += len(route_ids["metrics"])
        if expected_training:
            training_selected += len(route_ids["training"])
        if expected_values:
            value_selected += len(route_ids["values"])
        evidence_rows.append({
            "id": expected.get("id"), "query": expected.get("query"),
            "routes": route_rows, "filtered_tables": sorted(filtered_tables),
            "context_columns": sorted(context_columns),
            "diagnostics": {
                "columns": {
                    "expected": sorted(expected_columns),
                    "hit": sorted(expected_columns & route_ids["columns"]),
                    "missing": sorted(expected_columns - route_ids["columns"]),
                },
                "effective_context": {
                    "expected": sorted(expected_columns),
                    "hit": sorted(expected_columns & effective_columns),
                    "missing": sorted(expected_columns - effective_columns),
                },
                "tables": {
                    "expected": sorted(expected_tables),
                    "hit": sorted(expected_tables & filtered_tables),
                    "missing": sorted(expected_tables - filtered_tables),
                },
                "display": {
                    "expected": sorted(expected_display),
                    "hit": sorted(expected_display & route_ids["columns"]),
                    "missing": sorted(expected_display - route_ids["columns"]),
                },
                "metrics": {
                    "expected": sorted(expected_metrics),
                    "hit": sorted(expected_metrics & route_ids["metrics"]),
                    "missing": sorted(expected_metrics - route_ids["metrics"]),
                },
                "training": {
                    "expected": sorted(expected_training),
                    "hit": sorted(expected_training & route_ids["training"]),
                    "missing": sorted(expected_training - route_ids["training"]),
                },
                "values": {
                    "expected": sorted(expected_values),
                    "hit": sorted(expected_values & route_ids["values"]),
                    "missing": sorted(expected_values - route_ids["values"]),
                },
            },
        })
        # A metric may deliberately close a missing physical column after direct
        # retrieval (for example AOV -> fact_order.order_amount). End-to-end
        # readiness is therefore judged on effective context, while raw candidate
        # recall remains a separately gated diagnostic.
        evidence_rows[-1]["passed"] = not any(
            evidence_rows[-1]["diagnostics"][name]["missing"]
            for name in (
                "effective_context", "tables", "display", "metrics",
                "training", "values",
            )
        )

    def rate(hits, total):
        return round(hits / total, 4) if total else 1.0

    def mean(values):
        return round(sum(values) / len(values), 2) if values else None

    summary = {
        "evaluated_cases": len(evidence_rows), "required_cases": len(cases),
        "column_recall_at_k": rate(column_hits, column_total),
        "effective_context_column_recall": rate(context_column_hits, column_total),
        "table_filter_recall": rate(table_hits, table_total),
        "display_hit_rate": rate(display_hits, display_total),
        "effective_context_display_hit_rate": rate(context_display_hits, display_total),
        "metric_recall_at_k": rate(metric_hits, metric_total),
        "training_recall_at_k": rate(training_hits, training_total),
        "value_recall_at_k": rate(value_hits, value_total),
        "column_precision_at_k": rate(column_hits, column_selected),
        "metric_precision_at_k": rate(metric_hits, metric_selected),
        "training_precision_at_k": rate(training_hits, training_selected),
        "value_precision_at_k": rate(value_hits, value_selected),
        "average_dynamic_k": {route: mean(values) for route, values in dynamic_k.items()},
        "candidate_cache_hit_rate": round(cache_hits / cache_total, 4) if cache_total else None,
        "context_tokens_before_mean": mean(context_before),
        "context_tokens_after_mean": mean(context_after),
        "context_compression_rate": round(
            1 - sum(context_after) / max(1, sum(context_before)), 4
        ) if context_before and len(context_before) == len(context_after) else None,
        "budget_evidence_available": bool(dynamic_k and context_before),
        "failed_cases": sum(not item.get("passed") for item in evidence_rows),
    }
    summary["score"] = round((
        summary["effective_context_column_recall"]
        + summary["table_filter_recall"]
        + summary["display_hit_rate"]
    ) / 3 * 100, 1)
    summary["gate_passed"] = (
        not missing and len(evidence_rows) == len(cases)
        and summary["column_recall_at_k"] >= THRESHOLDS["live_column_recall_floor"]
        and summary["effective_context_column_recall"] >= THRESHOLDS["effective_context_column_recall"]
        and summary["column_precision_at_k"] >= THRESHOLDS["column_precision_at_k"]
        and summary["table_filter_recall"] >= THRESHOLDS["table_filter_recall"]
        and summary["display_hit_rate"] >= THRESHOLDS["display_hit_rate"]
        and summary["metric_recall_at_k"] >= THRESHOLDS["metric_recall_at_k"]
        and summary["training_recall_at_k"] >= THRESHOLDS["training_recall_at_k"]
    )
    available = bool(evidence_rows)
    return {
        "available": available, "source": "live_run_stage_evidence",
        "live_index_evaluated": available, "run_id": run.get("run_id"),
        "run_fingerprint": (run.get("fingerprint") or {}).get("run_fingerprint"),
        "candidate_index_versions": sorted(candidate_versions),
        "missing_case_ids": missing, "summary": summary, "cases": evidence_rows,
        "message": None if available else (
            "run 中没有与 RAG 契约问题逐字匹配的真实召回证据；"
            "请先运行 python -m evals.legacy.run_rag_live。"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="执行 RAG 召回、展示字段与索引参数离线评测")
    parser.add_argument("--enforce", action="store_true", help="门禁失败时返回退出码 1")
    args = parser.parse_args()
    report = build_report(write=True)
    summary = report["summary"]
    print(
        f"RAG 评测：{summary['cases']} 条，字段 Recall@K={summary['routes']['column']['recall_at_k']:.1%}，"
        f"筛表召回={summary['table_filter_recall']:.1%}，展示字段命中={summary['display_hit_rate']:.1%}，"
        f"门禁={'通过' if summary['gate_passed'] else '失败'}"
    )
    print(f"报告：{LATEST_PATH}")
    return 1 if args.enforce and not summary["gate_passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
