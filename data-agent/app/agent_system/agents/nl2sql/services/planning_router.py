"""TODO-63 versioned SQL-planning router and deterministic shadow candidates."""

from __future__ import annotations

import math
import re
import threading
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from app.agent_system.agents.nl2sql.settings import Nl2SqlSettings
from app.agent_system.agents.nl2sql.services.model_sql_policy import use_model_sql
from app.agent_system.agents.nl2sql.contracts.planning_route import (
    PLANNING_FEATURE_SCHEMA,
    PLANNING_ROUTE_CONTRACT_VERSION,
    PlanningRouteDecisionV1,
    PlanningRouteFeaturesV1,
)
from app.agent_system.agents.nl2sql.contracts.query_signals import QuerySignalsV1, stable_signal_hash


LEGACY_COMPLEX_KEYWORDS = (
    "占比",
    "比例",
    "趋势",
    "同比",
    "环比",
    "对比",
    "分布",
    "并且",
    "然后",
    "同时",
)
IMPLEMENTED_PROFILES = {
    "never_plan",
    "keyword_length_v1",
    "always_plan",
    "structured_rules_v1",
    "weighted_scorecard_v1",
    "calibrated_linear_v1",
}
# 这些 profile 对任意合法特征都给出终态动作，不会再次 abstain；只有它们
# 可以作为策略性 abstain 的安全 fallback。该集合只供证据治理校验，不改变
# 路由选择或 fallback 执行行为。
POLICY_FALLBACK_PROFILES = frozenset({
    "never_plan",
    "keyword_length_v1",
    "always_plan",
    "structured_rules_v1",
})
FEATURE_LEVELS = ("A0", "A1", "A2", "A3", "A4")

_runtime_lock = threading.RLock()
_runtime_profile_override: str | None = None

_DEFAULT_PLANNING = {
    "active_profile": "keyword_length_v1",
    "fallback_profile": "keyword_length_v1",
    "shadow_profiles": (
        "structured_rules_v1",
        "weighted_scorecard_v1",
        "calibrated_linear_v1",
    ),
    "repair_profile": "registry_same_rules_v1",
    "cache_namespace_version": "planning-route-cache-v1",
    "direct_threshold": 0.35,
    "plan_threshold": 0.62,
    "linear_direct_threshold": 0.35,
    "linear_plan_threshold": 0.62,
    "scorecard_weights": {
        "legacy_trigger": 0.22, "long_query": 0.18, "clause_count": 0.12,
        "signal_complexity": 0.18, "time_signal": 0.16, "numeric_signal": 0.12,
        "ir_fallback": 0.2, "having": 0.72, "anti_join": 0.78,
        "ratio": 0.42, "comparison": 0.36, "trend": 0.38,
        "distinct": 0.25, "ranking": 0.22, "window": 0.7,
        "table_structure": 0.34, "join_ambiguity": 0.62,
        "multi_measure": 0.48, "multi_condition": 0.25,
        "follow_up": 0.12, "budget_pressure": -0.08,
    },
    "linear_coefficients": {
        "intercept": -1.45, "legacy_trigger": 0.25, "long_query": -0.15,
        "signal_complexity": 0.35, "time_signal": 0.45, "numeric_signal": 0.3,
        "ir_fallback": 0.55, "having": 2.1, "anti_join": 2.25,
        "ratio": 1.15, "comparison": 0.95, "trend": 1.0,
        "distinct": 0.55, "ranking": 0.4, "window": 2.0,
        "table_structure": 0.85, "join_ambiguity": 1.55,
        "multi_measure": 1.2, "multi_condition": 0.55,
        "follow_up": 0.25, "budget_pressure": -0.35,
    },
}


def _planning_value(settings: Nl2SqlSettings | None, name: str) -> Any:
    section = settings.planning_route if settings is not None else None
    return _DEFAULT_PLANNING[name] if section is None else getattr(
        section, name, _DEFAULT_PLANNING[name]
    )


def _planning_timeout_seconds(settings: Nl2SqlSettings | None) -> float:
    section = settings.llm if settings is not None else None
    routes = getattr(section, "routes", {}) if section is not None else {}
    default = float(getattr(section, "timeout_seconds", 30.0)) if section is not None else 30.0
    return float((routes.get("planning") or {}).get("timeout_seconds", default))


def keyword_length_needs_plan(query: str) -> bool:
    """Character-for-character compatible C1 baseline."""

    if not query:
        return False
    if any(keyword in query for keyword in LEGACY_COMPLEX_KEYWORDS):
        return True
    return len(query) > 32


def runtime_profile_status(
    settings: Nl2SqlSettings | None = None,
) -> dict[str, Any]:
    with _runtime_lock:
        override = _runtime_profile_override
    configured = str(_planning_value(settings, "active_profile"))
    return {
        "configured_profile": configured,
        "runtime_profile": override or configured,
        "runtime_override": override,
        "fallback_profile": str(_planning_value(settings, "fallback_profile")),
        "rollback_without_restart": True,
    }


def rollback_to_fallback(
    settings: Nl2SqlSettings | None = None,
) -> dict[str, Any]:
    """Apply the pre-registered safe profile immediately in this process."""

    global _runtime_profile_override
    fallback = str(_planning_value(settings, "fallback_profile"))
    if fallback not in IMPLEMENTED_PROFILES:
        raise ValueError(f"未实现 fallback profile：{fallback}")
    with _runtime_lock:
        _runtime_profile_override = fallback
    return runtime_profile_status(settings)


def set_runtime_profile_for_test(profile: str | None) -> None:
    """Test-only deterministic override; production API only exposes rollback."""

    global _runtime_profile_override
    if profile is not None and profile not in IMPLEMENTED_PROFILES:
        raise ValueError(f"未实现 planning profile：{profile}")
    with _runtime_lock:
        _runtime_profile_override = profile


def _length_bucket(length: int) -> str:
    if length <= 12:
        return "0_12"
    if length <= 24:
        return "13_24"
    if length <= 32:
        return "25_32"
    if length <= 64:
        return "33_64"
    return "65_plus"


def _timeout_bucket(milliseconds: int) -> str:
    if milliseconds < 5000:
        return "under_5s"
    if milliseconds < 10000:
        return "under_10s"
    if milliseconds < 20000:
        return "under_20s"
    return "20s_plus"


def _ir_reason_code(reason: str) -> str:
    text = str(reason or "").lower()
    rules = (
        (("空问题",), "empty_query"),
        (("多轮", "省略"), "follow_up_ellipsis"),
        (("未完整召回", "缺表"), "missing_table"),
        (("没有唯一", "连接路径"), "ambiguous_join"),
        (("过滤", "时间", "比例", "比较"), "unsupported_semantics"),
        (("白名单",), "intent_not_whitelisted"),
        (("证据不足",), "insufficient_semantics"),
    )
    for tokens, code in rules:
        if any(token in text for token in tokens):
            return code
    return "unknown"


def _bounded_count(value: Any, maximum: int = 99) -> int:
    try:
        return max(0, min(maximum, int(value or 0)))
    except (TypeError, ValueError):
        return 0


def _preserved_contract_count(preserve: Mapping[str, Any]) -> int:
    """Count heterogeneous follow-up slots without assuming every value is sized."""

    count = 0
    for value in preserve.values():
        if value is None:
            continue
        if isinstance(value, bool):
            count += int(value)
        elif isinstance(value, (str, bytes)):
            count += int(bool(value))
        elif isinstance(value, Mapping):
            count += len(value)
        elif isinstance(value, Sequence):
            count += len(value)
        else:
            count += 1
    return count


def _join_structure(table_infos: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    names: list[str] = []
    for table in table_infos:
        for column in table.get("columns") or []:
            role = str(column.get("role") or "").lower()
            name = str(column.get("name") or "").lower()
            if role in {"primary_key", "foreign_key", "pk", "fk"} or (
                name.endswith("_id") or name == "id"
            ):
                names.append(name)
    counts = Counter(names)
    groups = sum(count >= 2 for count in counts.values())
    ambiguous = sum(count >= 3 for count in counts.values())
    return groups, ambiguous


def extract_planning_features(
    state: Mapping[str, Any],
    *,
    settings: Nl2SqlSettings | None = None,
) -> PlanningRouteFeaturesV1:
    query = str(state.get("query") or "")
    signals = QuerySignalsV1.from_state(state.get("query_signals"))
    semantic_ir = dict(state.get("semantic_ir") or {})
    compiler = dict(semantic_ir.get("compiler") or {})
    table_infos = [
        item for item in (state.get("table_infos") or [])
        if isinstance(item, Mapping)
    ]
    metric_infos = [
        item for item in (state.get("metric_infos") or [])
        if isinstance(item, Mapping)
    ]
    join_groups, ambiguous_groups = _join_structure(table_infos)
    table_roles = [
        str(item.get("role") or "").lower() for item in table_infos
    ]
    column_count = sum(
        len(item.get("columns") or []) for item in table_infos
    )
    follow_up = dict(state.get("follow_up") or {})
    preserve = follow_up.get("preserve") or {}
    inherited_count = (
        _preserved_contract_count(preserve)
        if isinstance(preserve, Mapping)
        else 0
    )
    timeout_ms = int(_planning_timeout_seconds(settings) * 1000)

    legacy_hits = [
        keyword for keyword in LEGACY_COMPLEX_KEYWORDS if keyword in query
    ]
    conjunction_count = sum(
        query.count(value) for value in ("并且", "然后", "同时", "以及", "且")
    )
    clause_count = len([
        item for item in re.split(r"[，,。；;！？!?]+", query) if item.strip()
    ])
    has_ratio = bool(re.search(r"占比|比例|比率|率(?:是多少|按|的|$)|客单价", query))
    has_comparison = bool(re.search(r"同比|环比|对比|相比|变化|增长|下降", query))
    has_trend = bool(re.search(r"趋势|逐月|每月|按月|按周|按季度|走势", query))
    has_having = bool(re.search(
        r"(累计|合计|总额|总数|平均|总消费|消费金额|订单金额).{0,12}"
        r"(超过|高于|大于|不少于|至少|低于|小于|不超过|至多)",
        query,
    ))
    has_anti_join = bool(re.search(
        r"没有|从未|未曾|未下单|未购买|未发生|不存在|not exists",
        query,
        flags=re.IGNORECASE,
    ))
    has_distinct = bool(re.search(r"去重|不同|唯一|独立客户", query))
    has_ranking = bool(re.search(r"前\s*\d+|后\s*\d+|最高|最低|排名|top\s*\d+", query, re.I))
    has_window = bool(re.search(r"最近一笔|最后一笔|每个.+最新|连续|累计排名|移动平均", query))
    measure_markers = re.findall(
        r"GMV|AOV|销售额|销售总额|订单数|客户数|客单价|复购率|"
        r"利润|收入|成本|销量|金额|数量|平均",
        query,
        flags=re.IGNORECASE,
    )
    has_multi_measure = (
        len({value.casefold() for value in measure_markers}) >= 2
        or len(semantic_ir.get("measures") or []) >= 2
        or len(metric_infos) >= 2
    )
    condition_markers = re.findall(
        r"超过|高于|低于|不少于|至少|不超过|并且|且|同时|"
        r"今年|去年|本月|上月|地区|品类",
        query,
    )
    has_multi_condition = len(condition_markers) >= 2 or clause_count >= 3
    has_aggregation = bool(re.search(
        r"总额|总数|合计|平均|多少|统计|累计|GMV|AOV|客单价|复购率|销量",
        query,
        flags=re.IGNORECASE,
    ))

    return PlanningRouteFeaturesV1(
        query_length_bucket=_length_bucket(len(query)),
        query_length_capped=min(128, len(query)),
        clause_count=min(12, max(1, clause_count)),
        legacy_keyword_count=min(10, len(legacy_hits)),
        conjunction_count=min(10, conjunction_count),
        time_signal_count=_bounded_count(len(signals.time_expressions)),
        numeric_signal_count=_bounded_count(len(signals.numeric_constraints)),
        operator_signal_count=_bounded_count(len(signals.operators)),
        polarity_signal_count=_bounded_count(len(signals.polarity)),
        identifier_signal_count=_bounded_count(len(signals.identifiers)),
        entity_signal_count=_bounded_count(len(signals.entity_literals)),
        ir_status="supported" if compiler.get("supported") else "unsupported",
        ir_intent=str(semantic_ir.get("intent") or "unknown")[:48],
        ir_reason_code=_ir_reason_code(str(compiler.get("reason") or "")),
        ir_dimension_count=_bounded_count(len(semantic_ir.get("dimensions") or [])),
        ir_measure_count=_bounded_count(len(semantic_ir.get("measures") or [])),
        ir_join_count=_bounded_count(len(semantic_ir.get("joins") or [])),
        table_count=_bounded_count(len(table_infos)),
        fact_table_count=_bounded_count(sum(role == "fact" for role in table_roles)),
        dimension_table_count=_bounded_count(sum(role == "dimension" for role in table_roles)),
        column_count=_bounded_count(column_count, 999),
        metric_count=_bounded_count(len(metric_infos)),
        join_key_group_count=_bounded_count(join_groups),
        ambiguous_join_group_count=_bounded_count(ambiguous_groups),
        has_aggregation=has_aggregation,
        has_ratio=has_ratio,
        has_comparison=has_comparison,
        has_trend=has_trend,
        has_having=has_having,
        has_anti_join=has_anti_join,
        has_distinct=has_distinct,
        has_ranking=has_ranking,
        has_window=has_window,
        has_multi_measure=has_multi_measure,
        has_multi_condition=has_multi_condition,
        is_follow_up=bool(
            follow_up.get("action")
            and follow_up.get("action") not in {"new_topic", "clarify"}
        ),
        inherited_contract_count=_bounded_count(inherited_count),
        planning_timeout_bucket=_timeout_bucket(timeout_ms),
    )


_FAMILY_MIN_LEVEL = {
    "legacy_trigger": "A0",
    "long_query": "A0",
    "clause_count": "A0",
    "signal_complexity": "A1",
    "time_signal": "A1",
    "numeric_signal": "A1",
    "ir_fallback": "A2",
    "having": "A2",
    "anti_join": "A2",
    "ratio": "A2",
    "comparison": "A2",
    "trend": "A2",
    "distinct": "A2",
    "ranking": "A2",
    "window": "A2",
    "table_structure": "A3",
    "join_ambiguity": "A3",
    "multi_measure": "A3",
    "multi_condition": "A3",
    "follow_up": "A4",
    "budget_pressure": "A4",
}


def planning_feature_vector(
    features: PlanningRouteFeaturesV1,
    feature_level: str = "A4",
) -> dict[str, float]:
    if feature_level not in FEATURE_LEVELS:
        raise ValueError(f"无效 feature level：{feature_level}")
    max_level = FEATURE_LEVELS.index(feature_level)
    raw = {
        "legacy_trigger": float(features.legacy_keyword_count > 0),
        "long_query": float(features.query_length_capped > 32),
        "clause_count": min(1.0, max(0, features.clause_count - 1) / 3),
        "signal_complexity": min(
            1.0,
            (
                features.operator_signal_count
                + features.polarity_signal_count
                + features.identifier_signal_count
            )
            / 3,
        ),
        "time_signal": float(
            features.time_signal_count > 0 or features.has_trend
        ),
        "numeric_signal": float(features.numeric_signal_count > 0),
        "ir_fallback": float(
            features.ir_reason_code
            in {"missing_table", "ambiguous_join", "unsupported_semantics"}
        ),
        "having": float(features.has_having),
        "anti_join": float(features.has_anti_join),
        "ratio": float(features.has_ratio),
        "comparison": float(features.has_comparison),
        "trend": float(features.has_trend),
        "distinct": float(features.has_distinct),
        "ranking": float(features.has_ranking),
        "window": float(features.has_window),
        "table_structure": min(
            1.0,
            (
                max(0, features.table_count - 1)
                + features.join_key_group_count
            )
            / 3,
        ),
        "join_ambiguity": float(features.ambiguous_join_group_count > 0),
        "multi_measure": float(features.has_multi_measure),
        "multi_condition": float(features.has_multi_condition),
        "follow_up": float(
            features.is_follow_up or features.inherited_contract_count > 0
        ),
        "budget_pressure": float(
            features.planning_timeout_bucket == "under_5s"
        ),
    }
    return {
        name: value
        for name, value in raw.items()
        if FEATURE_LEVELS.index(_FAMILY_MIN_LEVEL[name]) <= max_level
    }


def _decision(
    profile: str,
    features: PlanningRouteFeaturesV1,
    *,
    feature_level: str = "A4",
    linear_coefficients: Mapping[str, float] | None = None,
    settings: Nl2SqlSettings | None = None,
) -> tuple[str, float, float, tuple[str, ...]]:
    query_length = features.query_length_capped
    legacy = features.legacy_keyword_count > 0
    if profile == "never_plan":
        return "direct_generate", 0.0, 1.0, ("control_never_plan",)
    if profile == "always_plan":
        return "plan_then_generate", 1.0, 1.0, ("control_always_plan",)
    if profile == "keyword_length_v1":
        plan = legacy or query_length > 32
        reasons = []
        if legacy:
            reasons.append("legacy_complex_keyword")
        if query_length > 32:
            reasons.append("query_length_gt_32")
        if not reasons:
            reasons.append("legacy_no_trigger")
        return (
            "plan_then_generate" if plan else "direct_generate",
            1.0 if plan else 0.0,
            1.0,
            tuple(reasons),
        )

    vector = planning_feature_vector(features, feature_level)
    if profile == "structured_rules_v1":
        reasons = []
        for name in (
            "having",
            "anti_join",
            "window",
            "multi_measure",
            "join_ambiguity",
        ):
            if vector.get(name):
                reasons.append(f"structure:{name}")
        if vector.get("ratio") and (
            vector.get("table_structure")
            or vector.get("multi_measure")
            or vector.get("signal_complexity")
        ):
            reasons.append("structure:derived_ratio")
        if vector.get("comparison") and (
            vector.get("time_signal") or vector.get("multi_condition")
        ):
            reasons.append("structure:period_comparison")
        if vector.get("trend") and vector.get("time_signal"):
            reasons.append("structure:time_series")
        if (
            vector.get("table_structure", 0) >= 0.66
            and vector.get("multi_condition")
        ):
            reasons.append("structure:multi_table_contract")
        plan = bool(reasons)
        return (
            "plan_then_generate" if plan else "direct_generate",
            0.88 if plan else 0.16,
            0.88 if plan else 0.84,
            tuple(reasons or ("structured_no_trigger",)),
        )

    if profile == "weighted_scorecard_v1":
        weights = dict(_planning_value(settings, "scorecard_weights") or {})
        contributions = {
            name: value * float(weights.get(name, 0.0))
            for name, value in vector.items()
        }
        score = max(0.0, min(1.0, sum(contributions.values())))
        ranked = sorted(
            (
                (name, value)
                for name, value in contributions.items()
                if value > 0
            ),
            key=lambda row: (-row[1], row[0]),
        )
        reasons = tuple(f"score:{name}" for name, _ in ranked[:5])
        if score >= float(_planning_value(settings, "plan_threshold")):
            action = "plan_then_generate"
            confidence = min(1.0, .5 + score / 2)
        elif score <= float(_planning_value(settings, "direct_threshold")):
            action = "direct_generate"
            confidence = min(1.0, .5 + (1 - score) / 2)
        else:
            action = "abstain"
            confidence = max(0.0, 1 - abs(.5 - score) * 2)
        return action, score, confidence, reasons or ("score:no_trigger",)

    if profile == "calibrated_linear_v1":
        coefficients = dict(
            linear_coefficients
            or _planning_value(settings, "linear_coefficients")
            or {}
        )
        logit = float(coefficients.get("intercept", -1.0))
        for name, value in vector.items():
            logit += value * float(coefficients.get(name, 0.0))
        score = 1 / (1 + math.exp(-max(-30.0, min(30.0, logit))))
        direct_threshold = float(
            _planning_value(settings, "linear_direct_threshold")
        )
        plan_threshold = float(
            _planning_value(settings, "linear_plan_threshold")
        )
        if score >= plan_threshold:
            action = "plan_then_generate"
        elif score <= direct_threshold:
            action = "direct_generate"
        else:
            action = "abstain"
        ranked = sorted(
            (
                (name, abs(value * float(coefficients.get(name, 0.0))))
                for name, value in vector.items()
                if value
            ),
            key=lambda row: (-row[1], row[0]),
        )
        reasons = tuple(f"linear:{name}" for name, _ in ranked[:5])
        return (
            action,
            score,
            max(score, 1 - score),
            reasons or ("linear:intercept_only",),
        )
    raise ValueError(f"未实现 planning profile：{profile}")


def _cache_namespace(
    profile: str,
    feature_level: str,
    settings: Nl2SqlSettings | None = None,
) -> str:
    payload = {
        "namespace_version": _planning_value(settings, "cache_namespace_version"),
        "contract": PLANNING_ROUTE_CONTRACT_VERSION,
        "feature_schema": PLANNING_FEATURE_SCHEMA,
        "feature_level": feature_level,
        "profile": profile,
        "fallback": _planning_value(settings, "fallback_profile"),
        "repair_profile": _planning_value(settings, "repair_profile"),
        "plan_threshold": _planning_value(settings, "plan_threshold"),
        "direct_threshold": _planning_value(settings, "direct_threshold"),
    }
    return f"planning:{stable_signal_hash(payload)[:20]}"


def decide_planning_route(
    state: Mapping[str, Any],
    *,
    profile: str | None = None,
    feature_level: str = "A4",
    shadow_profiles: Sequence[str] | None = None,
    linear_coefficients: Mapping[str, float] | None = None,
    settings: Nl2SqlSettings | None = None,
) -> PlanningRouteDecisionV1:
    """Resolve active decision and non-calling shadow candidates."""

    started = time.perf_counter()
    features = extract_planning_features(state, settings=settings)
    semantic_ir_supported = bool(
        features.ir_status == "supported"
        and not use_model_sql(
            state.get("semantic_ir") or {},
            query_settings=settings.query if settings is not None else None,
        )
    )
    status = runtime_profile_status(settings)
    selected_profile = str(profile or status["runtime_profile"])
    fallback_profile = str(_planning_value(settings, "fallback_profile"))
    fallback_applied = False
    degraded = False
    budget = {
        "remaining_deadline_ms": None,
        "planning_timeout_ms": int(_planning_timeout_seconds(settings) * 1000),
        "planning_timeout_bucket": features.planning_timeout_bucket,
    }
    try:
        if selected_profile not in IMPLEMENTED_PROFILES:
            raise ValueError(f"未实现 planning profile：{selected_profile}")
        if semantic_ir_supported:
            decision = "direct_generate"
            actual = "deterministic_ir_bypass"
            score = 0.0
            confidence = 1.0
            reasons = ("deterministic_ir_supported",)
        else:
            decision, score, confidence, reasons = _decision(
                selected_profile,
                features,
                feature_level=feature_level,
                linear_coefficients=linear_coefficients,
                settings=settings,
            )
            if decision == "abstain":
                fallback_applied = True
                fallback_decision, _, _, fallback_reasons = _decision(
                    fallback_profile,
                    features,
                    feature_level=feature_level,
                    settings=settings,
                )
                actual = (
                    "plan_then_generate"
                    if fallback_decision == "plan_then_generate"
                    else "direct_generate"
                )
                reasons = (*reasons, "abstain_fallback", *fallback_reasons[:2])
            else:
                actual = decision
    except Exception as exc:
        degraded = True
        fallback_applied = True
        selected_profile = fallback_profile
        decision, score, confidence, reasons = _decision(
            fallback_profile,
            features,
            feature_level=feature_level,
            settings=settings,
        )
        actual = decision
        reasons = (
            "router_degraded",
            f"router_error:{type(exc).__name__}",
            *reasons,
        )

    shadows = []
    if not semantic_ir_supported:
        requested_shadows = (
            tuple(shadow_profiles)
            if shadow_profiles is not None
            else tuple(_planning_value(settings, "shadow_profiles") or ())
        )
        for shadow_profile in requested_shadows:
            if shadow_profile == selected_profile:
                continue
            try:
                shadow_decision, shadow_score, shadow_confidence, shadow_reasons = _decision(
                    str(shadow_profile),
                    features,
                    feature_level=feature_level,
                    settings=settings,
                )
            except ValueError:
                continue
            shadows.append({
                "profile": str(shadow_profile),
                "decision": shadow_decision,
                "score": round(shadow_score, 4),
                "confidence": round(shadow_confidence, 4),
                "reason_codes": list(shadow_reasons),
            })
    elapsed = (time.perf_counter() - started) * 1000
    return PlanningRouteDecisionV1(
        profile=selected_profile,
        decision=decision,
        actual_action=actual,
        score=score,
        confidence=confidence,
        reason_codes=tuple(reasons),
        features=features,
        semantic_ir_supported=semantic_ir_supported,
        budget_snapshot=budget,
        decision_ms=elapsed,
        fallback_profile=fallback_profile,
        fallback_applied=fallback_applied,
        router_degraded=degraded,
        cache_namespace=_cache_namespace(selected_profile, feature_level, settings),
        repair_profile=str(_planning_value(settings, "repair_profile")),
        shadow_decisions=tuple(shadows),
    )
