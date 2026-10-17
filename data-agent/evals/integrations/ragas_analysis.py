"""面向现有 DA 评测产物的隔离式 RAGAS 分析适配器。

本模块只读取冻结数据集和既有 Live 产物，不导入生产 ``app``，也不会在导入时
加载 RAGAS。RAGAS 依赖仅在显式创建默认 scorer 时加载，便于项目继续把它作为
临时评测依赖使用。
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import random
import re
import time
import unicodedata
import warnings
from collections import defaultdict
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from evals.artifacts import LocalArtifactStore


RETRIEVAL_ROUTES = (
    "table",
    "column_direct",
    "column_effective",
    "metric",
    "training",
    "value",
)
SUPPORTED_RAGAS_VERSION = "0.4.3"
LANGCHAIN_COMMUNITY_CONSTRAINT = "<0.4"
RAGAS_ISOLATED_INSTALL_SPEC = (
    'uv run --frozen --with "ragas==0.4.3" '
    '--with "langchain-community<0.4"'
)
DEFAULT_SQL_SAMPLE_QUOTAS: Mapping[str, Mapping[bool, int]] = {
    "easy": {True: 5, False: 5},
    "medium": {True: 5, False: 5},
    "hard": {True: 3, False: 7},
}


def _distribution_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _major_minor(version: str) -> tuple[int, int] | None:
    match = re.match(r"^(\d+)\.(\d+)", version)
    return (int(match.group(1)), int(match.group(2))) if match else None


def ragas_runtime_metadata(*, require_supported: bool = False) -> dict[str, object]:
    """记录冻结版本和已知 LangChain Community 兼容约束。"""

    ragas_version = _distribution_version("ragas")
    community_version = _distribution_version("langchain-community")
    community_major_minor = (
        _major_minor(community_version) if community_version is not None else None
    )
    compatible = ragas_version == SUPPORTED_RAGAS_VERSION and (
        community_major_minor is None or community_major_minor < (0, 4)
    )
    metadata = {
        "supported_ragas_version": SUPPORTED_RAGAS_VERSION,
        "installed_ragas_version": ragas_version,
        "langchain_community_constraint": LANGCHAIN_COMMUNITY_CONSTRAINT,
        "installed_langchain_community_version": community_version,
        "compatible": compatible,
        "isolated_install_spec": RAGAS_ISOLATED_INSTALL_SPEC,
    }
    if require_supported and not compatible:
        raise RuntimeError(
            "RAGAS 运行时不兼容；需要 ragas==0.4.3 且 langchain-community<0.4，"
            f"当前 ragas={ragas_version or '未安装'}，"
            f"langchain-community={community_version or '未安装'}"
        )
    return metadata


@dataclass(frozen=True, slots=True)
class RetrievalIdSample:
    """一条只在单一召回 route 内比较 ID 的 RAGAS 样本。"""

    case_id: str
    query: str
    route: str
    reference_ids: tuple[str, ...]
    retrieved_ids: tuple[str, ...]
    live_case_status: str = "matched"


@dataclass(frozen=True, slots=True)
class IdMetricScores:
    """RAGAS ID-based Precision/Recall 的最小返回合同。"""

    precision: float
    recall: float


@dataclass(frozen=True, slots=True)
class RetrievalScoreResult:
    """保留一条 retrieval 样本的确定性与 RAGAS 结果。"""

    case_id: str
    query: str
    route: str
    reference_ids: tuple[str, ...]
    retrieved_ids: tuple[str, ...]
    live_case_status: str
    status: str
    deterministic_precision: float
    deterministic_recall: float
    ragas_precision: float | None
    ragas_recall: float | None
    latency_ms: float
    error_type: str | None = None
    error_message: str | None = None


class RetrievalRagasScorer(Protocol):
    """可注入的异步 RAGAS retrieval scorer 边界。"""

    def __call__(self, sample: RetrievalIdSample) -> Awaitable[IdMetricScores]: ...


@dataclass(frozen=True, slots=True)
class SqlJudgeSample:
    """SQLSemanticEquivalence 的一条校准样本。"""

    run_id: str
    dataset_suite: str
    difficulty: str
    case_id: str
    question: str
    response_sql: str
    reference_sql: str
    reference_contexts: tuple[str, ...]
    primary_label: bool


@dataclass(frozen=True, slots=True)
class SqlSamplingResult:
    """稳定分层抽样及其不足证据。"""

    seed: int
    samples: tuple[SqlJudgeSample, ...]
    requested_counts: Mapping[str, int]
    selected_counts: Mapping[str, int]
    shortfalls: Mapping[str, int]
    ineligible_counts: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class SqlMetricScore:
    """RAGAS SQL metric 的最小返回合同。"""

    value: float
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class SqlJudgeScoreResult:
    """保留一条 SQL judge 调用的标签、理由、耗时和错误。"""

    run_id: str
    dataset_suite: str
    difficulty: str
    case_id: str
    primary_label: bool
    status: str
    value: float | None
    predicted_label: bool | None
    reason: str | None
    latency_ms: float
    error_type: str | None = None
    error_message: str | None = None


class SqlRagasScorer(Protocol):
    """可注入的异步 RAGAS SQLSemanticEquivalence scorer 边界。"""

    def __call__(self, sample: SqlJudgeSample) -> Awaitable[SqlMetricScore]: ...


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for raw in values:
        value = str(raw or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return tuple(result)


def deterministic_id_precision(
    reference_ids: Sequence[str], retrieved_ids: Sequence[str]
) -> float:
    """计算去重后的 ID Precision；没有召回项时返回 0。"""

    relevant = set(_ordered_unique(reference_ids))
    retrieved = _ordered_unique(retrieved_ids)
    if not retrieved:
        return 0.0
    return len(relevant.intersection(retrieved)) / len(retrieved)


def deterministic_id_recall(
    reference_ids: Sequence[str], retrieved_ids: Sequence[str]
) -> float:
    """计算去重后的 ID Recall；无正 qrel 属于调用合同错误。"""

    relevant = set(_ordered_unique(reference_ids))
    if not relevant:
        raise ValueError("ID Recall 需要至少一个正 qrel")
    retrieved = set(_ordered_unique(retrieved_ids))
    return len(relevant.intersection(retrieved)) / len(relevant)


def _load_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须是对象：{path}")
    return payload


def _normalized_question(value: object) -> str:
    return re.sub(r"\s+", "", str(value or "")).casefold()


def _positive_qrel(row: Mapping[str, object]) -> bool:
    try:
        return float(row.get("relevance_grade") or 0) > 0
    except (TypeError, ValueError):
        return False


def _qrel_route(document_id: str) -> str | None:
    prefix = document_id.split(":", 1)[0]
    return prefix if prefix in {"table", "column", "metric", "training", "value"} else None


def _with_prefix(prefix: str, value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text if text.startswith(f"{prefix}:") else f"{prefix}:{text}"


def _mapping_rows(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _ranked_rows(value: object) -> list[Mapping[str, object]]:
    rows = _mapping_rows(value)

    def rank_key(item: tuple[int, Mapping[str, object]]) -> tuple[float, int]:
        index, row = item
        rank = row.get("rank")
        if isinstance(rank, (int, float)) and not isinstance(rank, bool):
            return float(rank), index
        return math.inf, index

    return [row for _, row in sorted(enumerate(rows), key=rank_key)]


def _normalize_value(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip().casefold()


def _value_document_id(workspace_id: str, column_id: str, raw_value: str) -> str:
    digest = hashlib.sha256(
        "\x1f".join((workspace_id, column_id, _normalize_value(raw_value))).encode("utf-8")
    ).hexdigest()
    return f"value:{digest}"


def _value_reference_lookup(
    workspace_id: str, reference_ids: Sequence[str]
) -> dict[str, str]:
    """把去标识化 value hash 映射回已在 qrel 中审核的稳定 ID。"""

    lookup: dict[str, str] = {}
    for reference_id in reference_ids:
        remainder = reference_id.removeprefix("value:")
        if ":" not in remainder:
            continue
        column_id, raw_value = remainder.split(":", 1)
        lookup[_value_document_id(workspace_id, column_id, raw_value)] = reference_id
    return lookup


def _actual_route_ids(
    live_case: Mapping[str, object] | None,
    *,
    route: str,
    workspace_id: str,
    reference_ids: Sequence[str],
) -> tuple[str, ...]:
    if live_case is None:
        return ()
    stage = live_case.get("stage_evidence")
    if not isinstance(stage, Mapping):
        return ()
    retrieval = stage.get("retrieval")
    if not isinstance(retrieval, Mapping):
        retrieval = {}

    if route == "table":
        raw_tables = stage.get("filtered_tables")
        if not isinstance(raw_tables, Sequence) or isinstance(raw_tables, (str, bytes)):
            return ()
        return _ordered_unique([_with_prefix("table", item) for item in raw_tables])

    if route in {"column_direct", "column_effective"}:
        direct = [
            _with_prefix("column", row.get("id") or row.get("name"))
            for row in _ranked_rows(retrieval.get("columns"))
        ]
        if route == "column_direct":
            return _ordered_unique(direct)
        context = stage.get("context_columns")
        context_ids: list[str] = []
        if isinstance(context, Sequence) and not isinstance(context, (str, bytes)):
            for item in context:
                raw = item.get("id") if isinstance(item, Mapping) else item
                context_ids.append(_with_prefix("column", raw))
        return _ordered_unique([*direct, *context_ids])

    source_key = {"metric": "metrics", "training": "training", "value": "values"}[route]
    rows = _ranked_rows(retrieval.get(source_key))
    if route == "metric":
        return _ordered_unique(
            [_with_prefix("metric", row.get("name") or row.get("id")) for row in rows]
        )
    if route == "training":
        return _ordered_unique(
            [
                _with_prefix("training", row.get("question") or row.get("id"))
                for row in rows
            ]
        )

    value_lookup = _value_reference_lookup(workspace_id, reference_ids)
    normalized_values: list[str] = []
    for row in rows:
        actual_id = _with_prefix("value", row.get("id"))
        normalized_values.append(value_lookup.get(actual_id, actual_id))
    return _ordered_unique(normalized_values)


def build_retrieval_id_samples(
    cases: Sequence[Mapping[str, object]],
    qrels: Sequence[Mapping[str, object]],
    live_run: Mapping[str, object],
) -> tuple[RetrievalIdSample, ...]:
    """按 route 构造 ID 样本，并过滤当前 case 没有正 qrel 的 route。"""

    qrels_by_case: defaultdict[str, defaultdict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for qrel in qrels:
        if not _positive_qrel(qrel):
            continue
        case_id = str(qrel.get("case_id") or "").strip()
        document_id = str(qrel.get("document_id") or "").strip()
        route = _qrel_route(document_id)
        if case_id and document_id and route is not None:
            qrels_by_case[case_id][route].append(document_id)

    run_cases = _mapping_rows(live_run.get("cases"))
    by_id: dict[str, Mapping[str, object]] = {}
    by_query: dict[str, Mapping[str, object]] = {}
    for item in run_cases:
        case_id = str(item.get("rag_case_id") or item.get("id") or "").strip()
        if case_id:
            by_id.setdefault(case_id, item)
        query_key = _normalized_question(item.get("question"))
        if query_key:
            by_query.setdefault(query_key, item)

    workspace_id = str(live_run.get("workspace_id") or "").strip()
    samples: list[RetrievalIdSample] = []
    for case in cases:
        case_id = str(case.get("case_id") or case.get("id") or "").strip()
        query = str(case.get("query") or case.get("question") or "").strip()
        if not case_id:
            continue
        live_case = by_id.get(case_id)
        live_status = "matched"
        if live_case is not None and _normalized_question(live_case.get("question")) != _normalized_question(query):
            live_case = None
            live_status = "question_mismatch"
        if live_case is None and live_status == "matched":
            live_case = by_query.get(_normalized_question(query))
            live_status = "matched_by_question" if live_case is not None else "missing_live_case"
        if live_case is not None:
            stage = live_case.get("stage_evidence")
            if not isinstance(stage, Mapping) or not isinstance(stage.get("retrieval"), Mapping):
                live_status = "missing_retrieval_evidence"

        case_qrels = qrels_by_case.get(case_id, {})
        for route in RETRIEVAL_ROUTES:
            qrel_route = "column" if route.startswith("column_") else route
            reference_ids = _ordered_unique(case_qrels.get(qrel_route, ()))
            if not reference_ids:
                continue
            retrieved_ids = _actual_route_ids(
                live_case,
                route=route,
                workspace_id=workspace_id,
                reference_ids=reference_ids,
            )
            samples.append(
                RetrievalIdSample(
                    case_id=case_id,
                    query=query,
                    route=route,
                    reference_ids=reference_ids,
                    retrieved_ids=retrieved_ids,
                    live_case_status=live_status,
                )
            )
    return tuple(samples)


def load_retrieval_id_samples(
    cases_path: str | Path,
    qrels_path: str | Path,
    live_run_path: str | Path,
) -> tuple[RetrievalIdSample, ...]:
    """从冻结 CSV 和 Live JSON 加载 retrieval ID 样本。"""

    return build_retrieval_id_samples(
        _load_csv(cases_path),
        _load_csv(qrels_path),
        _load_json(live_run_path),
    )


def _bounded_score(value: object, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"{name} 必须位于 [0, 1]")
    return number


def _coerce_id_scores(value: object) -> IdMetricScores:
    if isinstance(value, IdMetricScores):
        return IdMetricScores(
            _bounded_score(value.precision, "precision"),
            _bounded_score(value.recall, "recall"),
        )
    if isinstance(value, Mapping):
        return IdMetricScores(
            _bounded_score(value.get("precision"), "precision"),
            _bounded_score(value.get("recall"), "recall"),
        )
    raise TypeError("retrieval scorer 必须返回 IdMetricScores 或同形 Mapping")


def create_ragas_id_scorer() -> RetrievalRagasScorer:
    """惰性创建 RAGAS IDBasedContextPrecision/Recall scorer。"""

    ragas_runtime_metadata(require_supported=True)
    try:
        # 0.4.3 仍公开导出两个 ID metric，但会发出迁移提示；适配器锁版后只
        # 定向抑制这两项告警，避免依赖未承诺的私有模块路径。
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*IDBasedContext(?:Precision|Recall).*deprecated.*",
                category=DeprecationWarning,
            )
            from ragas import SingleTurnSample
            from ragas.metrics import IDBasedContextPrecision, IDBasedContextRecall
    except ImportError as exc:  # 可选依赖缺失必须在显式调用时报告。
        raise RuntimeError(
            "缺少可选依赖 ragas；请在隔离环境安装固定版本后重试"
        ) from exc

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*IDBasedContext(?:Precision|Recall).*deprecated.*",
            category=DeprecationWarning,
        )
        precision_metric = IDBasedContextPrecision()
        recall_metric = IDBasedContextRecall()

    async def score(sample: RetrievalIdSample) -> IdMetricScores:
        ragas_sample = SingleTurnSample(
            retrieved_context_ids=list(sample.retrieved_ids),
            reference_context_ids=list(sample.reference_ids),
        )
        precision = await precision_metric.single_turn_ascore(ragas_sample)
        recall = await recall_metric.single_turn_ascore(ragas_sample)
        return IdMetricScores(float(precision), float(recall))

    return score


async def score_retrieval_samples(
    samples: Sequence[RetrievalIdSample],
    scorer: RetrievalRagasScorer | None = None,
) -> tuple[RetrievalScoreResult, ...]:
    """逐样本运行 RAGAS；错误被记录且不得改变输入输出基数。"""

    active_scorer = scorer or create_ragas_id_scorer()
    results: list[RetrievalScoreResult] = []
    for sample in samples:
        deterministic_precision = deterministic_id_precision(
            sample.reference_ids, sample.retrieved_ids
        )
        deterministic_recall = deterministic_id_recall(
            sample.reference_ids, sample.retrieved_ids
        )
        started = time.perf_counter()
        try:
            scores = _coerce_id_scores(await active_scorer(sample))
            result = RetrievalScoreResult(
                case_id=sample.case_id,
                query=sample.query,
                route=sample.route,
                reference_ids=sample.reference_ids,
                retrieved_ids=sample.retrieved_ids,
                live_case_status=sample.live_case_status,
                status="computed",
                deterministic_precision=deterministic_precision,
                deterministic_recall=deterministic_recall,
                ragas_precision=scores.precision,
                ragas_recall=scores.recall,
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
            )
        except Exception as exc:  # 单个评估器失败不能丢弃其样本。
            result = RetrievalScoreResult(
                case_id=sample.case_id,
                query=sample.query,
                route=sample.route,
                reference_ids=sample.reference_ids,
                retrieved_ids=sample.retrieved_ids,
                live_case_status=sample.live_case_status,
                status="error",
                deterministic_precision=deterministic_precision,
                deterministic_recall=deterministic_recall,
                ragas_precision=None,
                ragas_recall=None,
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
                error_type=type(exc).__name__,
                error_message=str(exc)[:1000],
            )
        results.append(result)
    if len(results) != len(samples):
        raise AssertionError("retrieval scorer 输出数与输入数不一致")
    return tuple(results)


def _macro(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def summarize_retrieval_results(
    results: Sequence[RetrievalScoreResult],
) -> dict[str, object]:
    """按 route 汇总宏平均，不跨 route 拼接 rank 或生成总分。"""

    grouped: defaultdict[str, list[RetrievalScoreResult]] = defaultdict(list)
    for result in results:
        grouped[result.route].append(result)
    routes: dict[str, object] = {}
    for route in RETRIEVAL_ROUTES:
        rows = grouped.get(route, [])
        if not rows:
            continue
        computed = [row for row in rows if row.status == "computed"]
        routes[route] = {
            "sample_count": len(rows),
            "computed_count": len(computed),
            "error_count": sum(row.status == "error" for row in rows),
            "live_evidence_issue_count": sum(
                row.live_case_status not in {"matched", "matched_by_question"}
                for row in rows
            ),
            "deterministic": {
                "precision_macro": _macro([row.deterministic_precision for row in rows]),
                "recall_macro": _macro([row.deterministic_recall for row in rows]),
            },
            "ragas": {
                "precision_macro": _macro(
                    [row.ragas_precision for row in computed if row.ragas_precision is not None]
                ),
                "recall_macro": _macro(
                    [row.ragas_recall for row in computed if row.ragas_recall is not None]
                ),
            },
        }
    return {"sample_count": len(results), "routes": routes}


def _quota_key(difficulty: str, label: bool) -> str:
    return f"{difficulty}:{'positive' if label else 'negative'}"


def _stable_rng(seed: int, difficulty: str, label: bool) -> random.Random:
    digest = hashlib.sha256(
        f"{seed}|{difficulty}|{int(label)}".encode("utf-8")
    ).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def stratified_sample_sql_cases(
    run_payloads: Sequence[Mapping[str, object]],
    *,
    seed: int = 20260827,
    quotas: Mapping[str, Mapping[bool, int]] | None = None,
    reference_contexts: Sequence[str] = (),
) -> SqlSamplingResult:
    """按难度和 ``result_equivalent`` 做稳定抽样，并记录不足与排除原因。"""

    active_quotas = quotas or DEFAULT_SQL_SAMPLE_QUOTAS
    candidates: defaultdict[tuple[str, bool], list[SqlJudgeSample]] = defaultdict(list)
    ineligible: defaultdict[str, int] = defaultdict(int)
    frozen_contexts = tuple(str(item) for item in reference_contexts if str(item).strip())

    for run in run_payloads:
        run_id = str(run.get("run_id") or "unknown-run")
        dataset_suite = str(run.get("dataset_suite") or "unknown-dataset")
        for case in _mapping_rows(run.get("cases")):
            difficulty = str(case.get("difficulty") or "").strip().lower()
            label = case.get("result_equivalent")
            if difficulty not in active_quotas:
                ineligible["unsupported_difficulty"] += 1
                continue
            if not isinstance(label, bool):
                ineligible["missing_primary_label"] += 1
                continue
            question = str(case.get("question") or "").strip()
            response_sql = str(case.get("gen_sql") or "").strip()
            reference_sql = str(case.get("expected_sql") or "").strip()
            if not question:
                ineligible["missing_question"] += 1
                continue
            if not response_sql:
                ineligible["missing_response_sql"] += 1
                continue
            if not reference_sql:
                ineligible["missing_reference_sql"] += 1
                continue
            candidates[(difficulty, label)].append(
                SqlJudgeSample(
                    run_id=run_id,
                    dataset_suite=dataset_suite,
                    difficulty=difficulty,
                    case_id=str(case.get("id") or "unknown-case"),
                    question=question,
                    response_sql=response_sql,
                    reference_sql=reference_sql,
                    reference_contexts=frozen_contexts,
                    primary_label=label,
                )
            )

    requested: dict[str, int] = {}
    selected: dict[str, int] = {}
    shortfalls: dict[str, int] = {}
    samples: list[SqlJudgeSample] = []
    for difficulty in ("easy", "medium", "hard"):
        difficulty_quotas = active_quotas.get(difficulty, {})
        for label in (True, False):
            requested_count = int(difficulty_quotas.get(label, 0))
            if requested_count < 0:
                raise ValueError("SQL 分层配额不能为负数")
            key = _quota_key(difficulty, label)
            requested[key] = requested_count
            rows = sorted(
                candidates.get((difficulty, label), []),
                key=lambda item: (item.run_id, item.case_id),
            )
            _stable_rng(seed, difficulty, label).shuffle(rows)
            chosen = rows[:requested_count]
            samples.extend(chosen)
            selected[key] = len(chosen)
            shortfalls[key] = requested_count - len(chosen)

    return SqlSamplingResult(
        seed=seed,
        samples=tuple(samples),
        requested_counts=requested,
        selected_counts=selected,
        shortfalls=shortfalls,
        ineligible_counts=dict(sorted(ineligible.items())),
    )


def load_and_sample_sql_cases(
    run_paths: Sequence[str | Path],
    *,
    seed: int = 20260827,
    quotas: Mapping[str, Mapping[bool, int]] | None = None,
    reference_contexts: Sequence[str] = (),
) -> tuple[SqlSamplingResult, tuple[dict[str, Any], ...]]:
    """读取三档 SQL Live JSON，并返回抽样和原始运行对象。"""

    runs = tuple(_load_json(path) for path in run_paths)
    return (
        stratified_sample_sql_cases(
            runs,
            seed=seed,
            quotas=quotas,
            reference_contexts=reference_contexts,
        ),
        runs,
    )


def _coerce_sql_score(value: object) -> SqlMetricScore:
    if isinstance(value, SqlMetricScore):
        return SqlMetricScore(_bounded_score(value.value, "value"), value.reason)
    if isinstance(value, Mapping):
        reason = value.get("reason")
        return SqlMetricScore(
            _bounded_score(value.get("value"), "value"),
            None if reason is None else str(reason),
        )
    raw_value = getattr(value, "value", None)
    if raw_value is not None:
        reason = getattr(value, "reason", None)
        return SqlMetricScore(
            _bounded_score(raw_value, "value"),
            None if reason is None else str(reason),
        )
    raise TypeError("SQL scorer 必须返回 SqlMetricScore、同形 Mapping 或 RAGAS MetricResult")


def create_ragas_sql_semantic_scorer(*, llm: object) -> SqlRagasScorer:
    """惰性创建现代 RAGAS ``SQLSemanticEquivalence`` scorer。"""

    ragas_runtime_metadata(require_supported=True)
    try:
        from ragas.metrics.collections import SQLSemanticEquivalence
    except ImportError as exc:  # 可选依赖缺失必须在显式调用时报告。
        raise RuntimeError(
            "缺少可选依赖 ragas；请在隔离环境安装固定版本后重试"
        ) from exc

    metric = SQLSemanticEquivalence(llm=llm)

    async def score(sample: SqlJudgeSample) -> SqlMetricScore:
        result = await metric.ascore(
            response=sample.response_sql,
            reference=sample.reference_sql,
            reference_contexts=list(sample.reference_contexts),
        )
        return _coerce_sql_score(result)

    return score


async def score_sql_samples(
    samples: Sequence[SqlJudgeSample],
    scorer: SqlRagasScorer,
    *,
    threshold: float = 0.5,
) -> tuple[SqlJudgeScoreResult, ...]:
    """逐条运行 SQL judge；错误和理由均保留，且严格保持样本数。"""

    if not 0 <= threshold <= 1:
        raise ValueError("threshold 必须位于 [0, 1]")
    results: list[SqlJudgeScoreResult] = []
    for sample in samples:
        started = time.perf_counter()
        try:
            score = _coerce_sql_score(await scorer(sample))
            result = SqlJudgeScoreResult(
                run_id=sample.run_id,
                dataset_suite=sample.dataset_suite,
                difficulty=sample.difficulty,
                case_id=sample.case_id,
                primary_label=sample.primary_label,
                status="computed",
                value=score.value,
                predicted_label=score.value >= threshold,
                reason=score.reason,
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
            )
        except Exception as exc:  # Judge 错误是评测结果的一部分。
            result = SqlJudgeScoreResult(
                run_id=sample.run_id,
                dataset_suite=sample.dataset_suite,
                difficulty=sample.difficulty,
                case_id=sample.case_id,
                primary_label=sample.primary_label,
                status="error",
                value=None,
                predicted_label=None,
                reason=None,
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
                error_type=type(exc).__name__,
                error_message=str(exc)[:1000],
            )
        results.append(result)
    if len(results) != len(samples):
        raise AssertionError("SQL scorer 输出数与输入数不一致")
    return tuple(results)


def _skipped_sql_results(
    samples: Sequence[SqlJudgeSample], reason: str
) -> tuple[SqlJudgeScoreResult, ...]:
    return tuple(
        SqlJudgeScoreResult(
            run_id=sample.run_id,
            dataset_suite=sample.dataset_suite,
            difficulty=sample.difficulty,
            case_id=sample.case_id,
            primary_label=sample.primary_label,
            status="skipped",
            value=None,
            predicted_label=None,
            reason=reason,
            latency_ms=0.0,
        )
        for sample in samples
    )


def wilson_interval(
    successes: int, total: int, *, z: float = 1.959963984540054
) -> tuple[float, float] | None:
    """计算二项比例的 Wilson 置信区间。"""

    if total <= 0:
        return None
    if successes < 0 or successes > total:
        raise ValueError("successes 必须位于 [0, total]")
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _sql_judge_group(rows: Sequence[SqlJudgeScoreResult]) -> dict[str, object]:
    computed = [
        row
        for row in rows
        if row.status == "computed" and row.predicted_label is not None
    ]
    tp = sum(row.primary_label and row.predicted_label is True for row in computed)
    fp = sum(not row.primary_label and row.predicted_label is True for row in computed)
    tn = sum(not row.primary_label and row.predicted_label is False for row in computed)
    fn = sum(row.primary_label and row.predicted_label is False for row in computed)
    total = len(computed)
    correct = tp + tn
    accuracy = correct / total if total else None
    interval = wilson_interval(correct, total)
    if total:
        actual_positive = tp + fn
        actual_negative = tn + fp
        predicted_positive = tp + fp
        predicted_negative = tn + fn
        expected = (
            actual_positive * predicted_positive + actual_negative * predicted_negative
        ) / (total * total)
        kappa = None if math.isclose(expected, 1.0) else (accuracy - expected) / (1 - expected)
    else:
        kappa = None
    return {
        "sample_count": len(rows),
        "computed_count": total,
        "error_count": sum(row.status == "error" for row in rows),
        "skipped_count": sum(row.status == "skipped" for row in rows),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "accuracy": None if accuracy is None else round(accuracy, 6),
        "cohen_kappa": None if kappa is None else round(kappa, 6),
        "accuracy_wilson_95": None
        if interval is None
        else {"low": round(interval[0], 6), "high": round(interval[1], 6)},
    }


def summarize_sql_judge_results(
    results: Sequence[SqlJudgeScoreResult],
) -> dict[str, object]:
    """汇总 SQL judge 与确定性 ``result_equivalent`` 的一致性。"""

    by_difficulty = {
        difficulty: _sql_judge_group(
            [row for row in results if row.difficulty == difficulty]
        )
        for difficulty in ("easy", "medium", "hard")
        if any(row.difficulty == difficulty for row in results)
    }
    return {"overall": _sql_judge_group(results), "by_difficulty": by_difficulty}


def _percentile(values: Sequence[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(ratio * len(ordered)) - 1)
    return ordered[index]


def _distribution(values: Sequence[float]) -> dict[str, object]:
    return {
        "observed_count": len(values),
        "mean": None if not values else round(sum(values) / len(values), 4),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "max": max(values) if values else None,
    }


def _count_and_rate(cases: Sequence[Mapping[str, object]], key: str) -> dict[str, object]:
    count = sum(case.get(key) is True for case in cases)
    return {"count": count, "rate": round(count / len(cases), 6) if cases else None}


def summarize_full_sql_runs(
    run_payloads: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """汇总完整 SQL Live 运行的确定性、安全、性能与成本证据。"""

    all_cases: list[Mapping[str, object]] = []
    per_run: list[dict[str, object]] = []
    cost_availability: list[bool] = []
    estimated_costs: list[float] = []
    currencies: set[str] = set()
    security_bypasses = infrastructure_errors = degraded_cases = 0

    for run in run_payloads:
        cases = _mapping_rows(run.get("cases"))
        all_cases.extend(cases)
        aggregate = run.get("aggregate")
        aggregate = aggregate if isinstance(aggregate, Mapping) else {}
        cost_available = aggregate.get("cost_estimate_available") is True
        cost_availability.append(cost_available)
        if cost_available and isinstance(aggregate.get("estimated_cost_total"), (int, float)):
            estimated_costs.append(float(aggregate["estimated_cost_total"]))
        currency = str(aggregate.get("cost_currency") or "").strip()
        if currency:
            currencies.add(currency)
        security_bypasses += int(aggregate.get("security_guard_bypass_count") or 0)
        infrastructure_errors += int(aggregate.get("infrastructure_errors") or 0)
        degraded_cases += int(aggregate.get("degraded_cases") or 0)
        per_run.append(
            {
                "run_id": str(run.get("run_id") or "unknown-run"),
                "dataset_suite": str(run.get("dataset_suite") or "unknown-dataset"),
                "n": len(cases),
                "valid": run.get("valid") is True,
                "base_run_valid": run.get("base_run_valid") is True,
                "pipeline_evidence_eligible": run.get("pipeline_evidence_eligible") is True,
                "provider_request_ids_available": run.get("provider_request_ids_available") is True,
                "result_equivalent": _count_and_rate(cases, "result_equivalent"),
                "exec_success": _count_and_rate(cases, "exec_success"),
                "syntax_valid": _count_and_rate(cases, "syntax_valid"),
                "security_guard_bypass_count": int(
                    aggregate.get("security_guard_bypass_count") or 0
                ),
                "latency_p50_ms": aggregate.get("latency_p50"),
                "latency_p95_ms": aggregate.get("latency_p95"),
                "tokens_mean": aggregate.get("tokens_mean"),
                "tokens_p95": aggregate.get("tokens_p95"),
                "cost_estimate_available": cost_available,
            }
        )

    latencies = [
        float(case.get("latency_ms"))
        for case in all_cases
        if isinstance(case.get("latency_ms"), (int, float))
        and not isinstance(case.get("latency_ms"), bool)
    ]
    tokens = [
        float(case.get("total_tokens"))
        for case in all_cases
        if isinstance(case.get("total_tokens"), (int, float))
        and not isinstance(case.get("total_tokens"), bool)
    ]
    all_costs_available = bool(run_payloads) and all(cost_availability)
    cost_currency = next(iter(currencies)) if len(currencies) == 1 else None
    deterministic_keys = (
        "result_evaluated",
        "syntax_valid",
        "exec_success",
        "result_equivalent",
        "structure_equivalent",
        "tables_correct",
        "table_scope_safe",
    )
    return {
        "run_count": len(run_payloads),
        "case_count": len(all_cases),
        "per_run": per_run,
        "overall": {
            "deterministic": {
                key: _count_and_rate(all_cases, key) for key in deterministic_keys
            },
            "safety": {
                "security_guard_bypass_count": security_bypasses,
                "table_scope_unsafe_count": sum(
                    case.get("table_scope_safe") is False for case in all_cases
                ),
                "infrastructure_error_count": infrastructure_errors,
                "degraded_case_count": degraded_cases,
            },
            "latency_ms": _distribution(latencies),
            "tokens": {
                **_distribution(tokens),
                "total": int(sum(tokens)) if tokens else None,
            },
            "cost": {
                "available": all_costs_available,
                "estimated_total": round(sum(estimated_costs), 8)
                if all_costs_available
                else None,
                "currency": cost_currency,
                "unavailable_run_ids": [
                    item["run_id"]
                    for item, available in zip(per_run, cost_availability, strict=True)
                    if not available
                ],
            },
            "governance": {
                "all_base_runs_valid": bool(run_payloads)
                and all(run.get("base_run_valid") is True for run in run_payloads),
                "all_runs_valid": bool(run_payloads)
                and all(run.get("valid") is True for run in run_payloads),
                "all_pipeline_evidence_eligible": bool(run_payloads)
                and all(
                    run.get("pipeline_evidence_eligible") is True for run in run_payloads
                ),
                "all_provider_request_ids_available": bool(run_payloads)
                and all(
                    run.get("provider_request_ids_available") is True
                    for run in run_payloads
                ),
            },
        },
    }


async def run_ragas_analysis(
    *,
    cases_path: str | Path,
    qrels_path: str | Path,
    rag_live_run_path: str | Path,
    sql_run_paths: Sequence[str | Path],
    retrieval_scorer: RetrievalRagasScorer | None = None,
    sql_scorer: SqlRagasScorer | None = None,
    sql_reference_contexts: Sequence[str] = (),
    seed: int = 20260827,
    quotas: Mapping[str, Mapping[bool, int]] | None = None,
) -> dict[str, object]:
    """完成 retrieval 与 SQL 两条 RAGAS 分析路径的只读编排。"""

    retrieval_samples = load_retrieval_id_samples(
        cases_path, qrels_path, rag_live_run_path
    )
    retrieval_results = await score_retrieval_samples(
        retrieval_samples, retrieval_scorer
    )
    sql_sampling, sql_runs = load_and_sample_sql_cases(
        sql_run_paths,
        seed=seed,
        quotas=quotas,
        reference_contexts=sql_reference_contexts,
    )
    if sql_scorer is None:
        sql_results = _skipped_sql_results(
            sql_sampling.samples, "sql_ragas_scorer_not_configured"
        )
    else:
        sql_results = await score_sql_samples(sql_sampling.samples, sql_scorer)

    return {
        "schema_version": "ragas-da-analysis-v1",
        "ragas_runtime": ragas_runtime_metadata(),
        "retrieval": {
            "samples": [asdict(item) for item in retrieval_samples],
            "results": [asdict(item) for item in retrieval_results],
            "summary": summarize_retrieval_results(retrieval_results),
        },
        "sql": {
            "sampling": asdict(sql_sampling),
            "judge_results": [asdict(item) for item in sql_results],
            "judge_summary": summarize_sql_judge_results(sql_results),
            "full_run_summary": summarize_full_sql_runs(sql_runs),
        },
        "limitations": [
            "retrieval qrels 仍为 draft，只能作为诊断证据",
            "SQL RAGAS judge 以确定性 result_equivalent 为主标签进行校准",
            "未计算跨 retrieval、SQL、性能与安全维度的综合总分",
        ],
    }


_CREATE_TABLE_RE = re.compile(
    r"^[ \t]*CREATE[ \t]+TABLE[ \t]+(?:IF[ \t]+NOT[ \t]+EXISTS[ \t]+)?[^;]+;",
    flags=re.IGNORECASE | re.DOTALL | re.MULTILINE,
)


def extract_schema_ddl(
    ddl_path: str | Path, *, table_limit: int = 5
) -> tuple[str, ...]:
    """只提取前若干个永久 ``CREATE TABLE``，拒绝任何行值语句。"""

    if table_limit < 1:
        raise ValueError("table_limit 必须大于 0")
    source = Path(ddl_path).read_text(encoding="utf-8")
    statements = [match.group(0).strip() for match in _CREATE_TABLE_RE.finditer(source)]
    selected = statements[:table_limit]
    if len(selected) != table_limit:
        raise ValueError(
            f"DDL 需要至少 {table_limit} 个非 TEMPORARY CREATE TABLE，实际 {len(selected)} 个"
        )
    for statement in selected:
        upper = statement.upper()
        if "CREATE TEMPORARY TABLE" in upper or re.search(r"\bINSERT\b", upper):
            raise ValueError("DDL 上下文不得包含 TEMPORARY TABLE、INSERT 或行值")
    return tuple(selected)


def create_openai_compatible_sql_scorer(
    *,
    api_key: str,
    base_url: str,
    model: str,
    max_tokens: int = 4096,
    timeout_seconds: float = 120.0,
) -> tuple[SqlRagasScorer, object]:
    """用 OpenAI-compatible client 构造 SQL judge，并返回可关闭的 client。"""

    if not api_key.strip():
        raise ValueError("LLM_API_KEY 未配置")
    if max_tokens < 1:
        raise ValueError("max_tokens 必须大于 0")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds 必须大于 0")
    ragas_runtime_metadata(require_supported=True)
    try:
        from openai import AsyncOpenAI
        from ragas.llms.base import llm_factory
    except ImportError as exc:
        raise RuntimeError("RAGAS SQL judge 运行时依赖不完整") from exc
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout_seconds,
    )
    # RAGAS 0.4.3 的 Instructor 默认仅允许 1024 个输出 token；较复杂 SQL
    # 会在结构化 reason 完成前被截断。显式上限属于评估器配置，不改变被测系统。
    llm = llm_factory(
        model,
        provider="openai",
        client=client,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return create_ragas_sql_semantic_scorer(llm=llm), client


def _csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    if not rows:
        return b""
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        encoded = {
            key: json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            if isinstance(value, (dict, list, tuple))
            else ""
            if value is None
            else value
            for key, value in row.items()
        }
        writer.writerow(encoded)
    return buffer.getvalue().encode("utf-8-sig")


def write_analysis_artifacts(
    report: Mapping[str, object],
    *,
    output_root: str | Path,
    evaluation_id: str,
    run_id: str,
) -> dict[str, object]:
    """用安全 ArtifactStore 写入主报告和两类逐样本 CSV。"""

    retrieval = report.get("retrieval")
    sql = report.get("sql")
    retrieval = retrieval if isinstance(retrieval, Mapping) else {}
    sql = sql if isinstance(sql, Mapping) else {}
    retrieval_rows = _mapping_rows(retrieval.get("results"))
    sql_rows = _mapping_rows(sql.get("judge_results"))
    store = LocalArtifactStore(output_root)
    prefix = f"{run_id}"
    return {
        "analysis": store.write_json(
            evaluation_id, f"{prefix}/analysis.json", dict(report)
        ),
        "retrieval_csv": store.write_bytes(
            evaluation_id,
            f"{prefix}/retrieval-results.csv",
            _csv_bytes(retrieval_rows),
            media_type="text/csv",
        ),
        "sql_judge_csv": store.write_bytes(
            evaluation_id,
            f"{prefix}/sql-judge-results.csv",
            _csv_bytes(sql_rows),
            media_type="text/csv",
        ),
    }


def _latest_artifact(directory: Path, pattern: str) -> Path:
    paths = sorted(directory.glob(pattern), key=lambda path: (path.stat().st_mtime_ns, path.name))
    if not paths:
        raise FileNotFoundError(f"没有找到评测产物：{directory / pattern}")
    return paths[-1]


def _default_sql_run_paths() -> tuple[Path, ...]:
    directory = Path("var") / "evals" / "legacy" / "runs"
    return tuple(
        _latest_artifact(directory, f"run_{difficulty}_sql_v1_*.json")
        for difficulty in ("easy", "medium", "hard")
    )


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="用 RAGAS 对 DA 的 retrieval 与 SQL Live 产物做隔离分析"
    )
    parser.add_argument(
        "--cases",
        default="evals/datasets/rag_v1/cases.csv",
        help="RAG cases.csv 路径",
    )
    parser.add_argument(
        "--qrels",
        default="evals/datasets/rag_v1/relevance-judgments.csv",
        help="RAG relevance-judgments.csv 路径",
    )
    parser.add_argument(
        "--rag-run",
        help="fresh RAG Live JSON；省略时选 var/evals/legacy/rag_runs 中最新文件",
    )
    parser.add_argument(
        "--sql-runs",
        nargs=3,
        metavar=("EASY_JSON", "MEDIUM_JSON", "HARD_JSON"),
        help="easy、medium、hard 三档 SQL Live JSON；省略时分别选最新文件",
    )
    parser.add_argument(
        "--ddl", default="docker_windows/mysql/dw.sql", help="只读取表结构的 DDL 文件"
    )
    parser.add_argument("--output-root", default="var/evals")
    parser.add_argument("--evaluation-id", default="ragas-da-v1")
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument(
        "--sql-judge",
        action="store_true",
        help="启用会产生 LLM 调用的 RAGAS SQLSemanticEquivalence",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LLM_MODEL_NAME") or "deepseek-v4-pro",
        help="SQL judge 模型名",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLM_BASE_URL") or "https://api.deepseek.com",
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--judge-max-tokens",
        type=int,
        default=4096,
        help="RAGAS SQL Judge 的结构化输出 token 上限",
    )
    parser.add_argument(
        "--judge-timeout-seconds",
        type=float,
        default=120.0,
        help="RAGAS SQL Judge 的单次请求超时",
    )
    return parser


async def _execute_cli(args: argparse.Namespace) -> tuple[dict[str, object], tuple[Path, ...], Path]:
    rag_run_path = (
        Path(args.rag_run)
        if args.rag_run
        else _latest_artifact(
            Path("var") / "evals" / "legacy" / "rag_runs", "rag_run_*.json"
        )
    )
    sql_run_paths = (
        tuple(Path(path) for path in args.sql_runs)
        if args.sql_runs
        else _default_sql_run_paths()
    )
    contexts = extract_schema_ddl(args.ddl)
    sql_scorer: SqlRagasScorer | None = None
    client: object | None = None
    if args.sql_judge:
        try:
            from dotenv import load_dotenv
        except ImportError as exc:
            raise RuntimeError("缺少 python-dotenv，无法安全加载 .env") from exc
        # 独立模块或 STDIN 启动时，find_dotenv() 未必能得到可靠调用路径。
        # 显式绑定项目本地 secret 文件；已有进程变量仍优先，且凭据不进入报告。
        load_dotenv(dotenv_path=Path(".env"), override=False)
        api_key = os.getenv("LLM_API_KEY") or ""
        sql_scorer, client = create_openai_compatible_sql_scorer(
            api_key=api_key,
            base_url=args.base_url,
            model=args.model,
            max_tokens=args.judge_max_tokens,
            timeout_seconds=args.judge_timeout_seconds,
        )
    try:
        report = await run_ragas_analysis(
            cases_path=args.cases,
            qrels_path=args.qrels,
            rag_live_run_path=rag_run_path,
            sql_run_paths=sql_run_paths,
            sql_scorer=sql_scorer,
            sql_reference_contexts=contexts,
            seed=args.seed,
        )
        report["evaluator_config"] = {
            "sql_judge_enabled": bool(args.sql_judge),
            "judge_model": args.model if args.sql_judge else None,
            "judge_temperature": 0.0 if args.sql_judge else None,
            "judge_max_tokens": args.judge_max_tokens if args.sql_judge else None,
            "judge_timeout_seconds": args.judge_timeout_seconds
            if args.sql_judge
            else None,
            "thinking_mode": "provider_default_not_overridden"
            if args.sql_judge
            else None,
        }
    finally:
        if client is not None:
            close = getattr(client, "close", None)
            if callable(close):
                await close()
    return report, sql_run_paths, rag_run_path


def main(argv: Sequence[str] | None = None) -> int:
    """运行 CLI，并只把摘要与产物路径写到 stdout。"""

    args = _build_cli_parser().parse_args(argv)
    report, sql_run_paths, rag_run_path = asyncio.run(_execute_cli(args))
    run_id = datetime.now(UTC).strftime("run-%Y%m%dT%H%M%S%fZ")
    report["run_id"] = run_id
    report["sources"] = {
        "cases": str(Path(args.cases).resolve()),
        "qrels": str(Path(args.qrels).resolve()),
        "rag_live_run": str(rag_run_path.resolve()),
        "sql_live_runs": [str(path.resolve()) for path in sql_run_paths],
        "ddl": str(Path(args.ddl).resolve()),
    }
    refs = write_analysis_artifacts(
        report,
        output_root=args.output_root,
        evaluation_id=args.evaluation_id,
        run_id=run_id,
    )

    retrieval_summary = report["retrieval"]["summary"]
    judge_overall = report["sql"]["judge_summary"]["overall"]
    deterministic = report["sql"]["full_run_summary"]["overall"]["deterministic"]
    result_equivalent = deterministic["result_equivalent"]
    print(
        "RAGAS DA 分析完成："
        f"retrieval={retrieval_summary['sample_count']} 条；"
        f"SQL Judge computed={judge_overall['computed_count']}、error={judge_overall['error_count']}；"
        f"Full SQL result_equivalent={result_equivalent['count']}/"
        f"{report['sql']['full_run_summary']['case_count']}"
    )
    for name in ("analysis", "retrieval_csv", "sql_judge_csv"):
        ref = refs[name]
        print(f"{name}: {ref.uri}")
    return 0


__all__ = [
    "DEFAULT_SQL_SAMPLE_QUOTAS",
    "LANGCHAIN_COMMUNITY_CONSTRAINT",
    "RAGAS_ISOLATED_INSTALL_SPEC",
    "SUPPORTED_RAGAS_VERSION",
    "IdMetricScores",
    "RetrievalIdSample",
    "RetrievalRagasScorer",
    "RetrievalScoreResult",
    "SqlJudgeSample",
    "SqlJudgeScoreResult",
    "SqlMetricScore",
    "SqlRagasScorer",
    "SqlSamplingResult",
    "build_retrieval_id_samples",
    "create_openai_compatible_sql_scorer",
    "create_ragas_id_scorer",
    "create_ragas_sql_semantic_scorer",
    "deterministic_id_precision",
    "deterministic_id_recall",
    "extract_schema_ddl",
    "load_and_sample_sql_cases",
    "load_retrieval_id_samples",
    "ragas_runtime_metadata",
    "run_ragas_analysis",
    "score_retrieval_samples",
    "score_sql_samples",
    "stratified_sample_sql_cases",
    "summarize_full_sql_runs",
    "summarize_retrieval_results",
    "summarize_sql_judge_results",
    "wilson_interval",
    "write_analysis_artifacts",
]


if __name__ == "__main__":
    raise SystemExit(main())
