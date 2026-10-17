"""查询意图安全评测集的稳定数据合同。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from evals.contracts.models import ContractError, validate_identifier


class QueryIntentCategory(StrEnum):
    """意图节点允许输出的封闭分类。"""

    BUSINESS_QUERY = "business_query"
    BUSINESS_HELP = "business_help"
    OUT_OF_SCOPE = "out_of_scope"
    FORBIDDEN_OPERATION = "forbidden_operation"
    AMBIGUOUS = "ambiguous"


class QueryIntentDecision(StrEnum):
    """意图节点对 LangGraph 的路由决定。"""

    CONTINUE = "continue"
    RESPOND = "respond"


class QueryIntentResponseKind(StrEnum):
    """提前终止时向用户返回的响应种类。"""

    GUIDANCE = "guidance"
    REFUSAL = "refusal"
    CLARIFICATION = "clarification"


class QueryIntentReasonCode(StrEnum):
    """评测使用的封闭原因码，不接受自由文本标签。"""

    BUSINESS_QUERY = "business_query"
    BUSINESS_HELP = "business_help"
    OUT_OF_SCOPE = "out_of_scope"
    FORBIDDEN_OPERATION = "forbidden_operation"
    PROMPT_INJECTION = "prompt_injection"
    AMBIGUOUS = "ambiguous"


class QueryIntentSlice(StrEnum):
    """开发集预注册的分析切片。"""

    BUSINESS_QUERY = "business_query"
    BUSINESS_HELP = "business_help"
    OUT_OF_SCOPE = "out_of_scope"
    FORBIDDEN_OPERATION = "forbidden_operation"
    AMBIGUOUS = "ambiguous"
    CURRENT_TURN_PROMPT_INJECTION = "current_turn_prompt_injection"
    MIXED = "mixed"
    DESCRIPTIVE_WRITE_TERMS = "descriptive_write_terms"
    FOLLOW_UP = "follow_up"
    UPLOAD_SCHEMA = "upload_schema"
    NEAR_MISS = "near_miss"


class QueryIntentCaseOrigin(StrEnum):
    """样本来源；已知线上失败与合成开发样本必须可区分。"""

    SYNTHETIC = "synthetic"
    DEVELOPMENT_KNOWN_FAILURE = "development_known_failure"


class QueryIntentHistoryRole(StrEnum):
    """进入分类器前可见的历史消息角色。"""

    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class QueryIntentHistoryTurn:
    """一条最小历史消息，不承载数据库结果行。"""

    role: QueryIntentHistoryRole
    content: str

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ContractError("历史消息 content 不能为空")


@dataclass(frozen=True, slots=True)
class QueryIntentExpectedOutcome:
    """分类、终止行为和 SQL 生成权限的联合 Gold 合同。"""

    category: QueryIntentCategory
    decision: QueryIntentDecision
    response_kind: QueryIntentResponseKind | None
    reason_code: QueryIntentReasonCode
    terminal: bool
    sql_generation_allowed: bool

    def __post_init__(self) -> None:
        expected_by_category = {
            QueryIntentCategory.BUSINESS_QUERY: (
                QueryIntentDecision.CONTINUE,
                None,
                False,
                True,
            ),
            QueryIntentCategory.BUSINESS_HELP: (
                QueryIntentDecision.RESPOND,
                QueryIntentResponseKind.GUIDANCE,
                True,
                False,
            ),
            QueryIntentCategory.OUT_OF_SCOPE: (
                QueryIntentDecision.RESPOND,
                QueryIntentResponseKind.GUIDANCE,
                True,
                False,
            ),
            QueryIntentCategory.FORBIDDEN_OPERATION: (
                QueryIntentDecision.RESPOND,
                QueryIntentResponseKind.REFUSAL,
                True,
                False,
            ),
            QueryIntentCategory.AMBIGUOUS: (
                QueryIntentDecision.RESPOND,
                QueryIntentResponseKind.CLARIFICATION,
                True,
                False,
            ),
        }
        actual = (
            self.decision,
            self.response_kind,
            self.terminal,
            self.sql_generation_allowed,
        )
        if actual != expected_by_category[self.category]:
            raise ContractError(
                f"{self.category.value} 的 decision/response/terminal/SQL 合同不一致"
            )

        allowed_reasons = {
            QueryIntentCategory.BUSINESS_QUERY: {
                QueryIntentReasonCode.BUSINESS_QUERY,
            },
            QueryIntentCategory.BUSINESS_HELP: {
                QueryIntentReasonCode.BUSINESS_HELP,
            },
            QueryIntentCategory.OUT_OF_SCOPE: {
                QueryIntentReasonCode.OUT_OF_SCOPE,
            },
            QueryIntentCategory.FORBIDDEN_OPERATION: {
                QueryIntentReasonCode.FORBIDDEN_OPERATION,
                QueryIntentReasonCode.PROMPT_INJECTION,
            },
            QueryIntentCategory.AMBIGUOUS: {
                QueryIntentReasonCode.AMBIGUOUS,
            },
        }
        if self.reason_code not in allowed_reasons[self.category]:
            raise ContractError(
                f"reason_code={self.reason_code.value} 与 category={self.category.value} 不一致"
            )


@dataclass(frozen=True, slots=True)
class QueryIntentSafetyCase:
    """一个查询意图安全开发样本。"""

    case_id: str
    dataset_id: str
    dataset_version: str
    split: str
    origin: QueryIntentCaseOrigin
    query: str
    history: tuple[QueryIntentHistoryTurn, ...]
    slices: tuple[QueryIntentSlice, ...]
    expected: QueryIntentExpectedOutcome
    known_failure_reference: str | None

    def __post_init__(self) -> None:
        validate_identifier("case_id", self.case_id)
        validate_identifier("dataset_id", self.dataset_id)
        validate_identifier("dataset_version", self.dataset_version)
        if self.split != "development":
            raise ContractError("query-intent-safety-v1 只包含 development 样本")
        if not self.query.strip():
            raise ContractError("query 不能为空")
        if not self.slices:
            raise ContractError("slices 至少包含主分类切片")
        if len(set(self.slices)) != len(self.slices):
            raise ContractError("slices 不能包含重复值")
        category_slice = QueryIntentSlice(self.expected.category.value)
        if category_slice not in self.slices:
            raise ContractError("slices 必须包含与 expected.category 同名的主切片")

        if self.history:
            if len(self.history) % 2 != 0:
                raise ContractError("history 必须由完整的 user/assistant 对组成")
            for index, turn in enumerate(self.history):
                expected_role = (
                    QueryIntentHistoryRole.USER
                    if index % 2 == 0
                    else QueryIntentHistoryRole.ASSISTANT
                )
                if turn.role is not expected_role:
                    raise ContractError("history 必须从 user 开始并严格交替")

        is_known_failure = (
            self.origin is QueryIntentCaseOrigin.DEVELOPMENT_KNOWN_FAILURE
        )
        if is_known_failure != bool(
            self.known_failure_reference and self.known_failure_reference.strip()
        ):
            raise ContractError(
                "development_known_failure 与 known_failure_reference 必须同时出现"
            )


@dataclass(frozen=True, slots=True)
class QueryIntentSafetyManifest:
    """版本化开发集清单。"""

    schema_version: str
    dataset_id: str
    dataset_version: str
    split: str
    case_file: str
    contains_hidden_labels: bool
    known_failure_case_ids: tuple[str, ...]
    required_slices: tuple[QueryIntentSlice, ...]

    def __post_init__(self) -> None:
        validate_identifier("schema_version", self.schema_version)
        validate_identifier("dataset_id", self.dataset_id)
        validate_identifier("dataset_version", self.dataset_version)
        if self.split != "development":
            raise ContractError("公开开发集 manifest 的 split 必须是 development")
        if self.case_file != "cases.jsonl":
            raise ContractError("case_file 必须是 cases.jsonl")
        if self.contains_hidden_labels:
            raise ContractError("公开开发集不能声明或包含 hidden labels")
        if len(set(self.known_failure_case_ids)) != len(self.known_failure_case_ids):
            raise ContractError("known_failure_case_ids 不能重复")
        for case_id in self.known_failure_case_ids:
            validate_identifier("known_failure_case_id", case_id)
        if not self.required_slices:
            raise ContractError("required_slices 不能为空")
        if len(set(self.required_slices)) != len(self.required_slices):
            raise ContractError("required_slices 不能重复")


@dataclass(frozen=True, slots=True)
class QueryIntentSafetyDataset:
    """已通过版本、一致性与切片覆盖校验的开发集。"""

    manifest: QueryIntentSafetyManifest
    cases: tuple[QueryIntentSafetyCase, ...]

    def __post_init__(self) -> None:
        if not self.cases:
            raise ContractError("查询意图安全开发集至少需要一个样本")
        case_ids: set[str] = set()
        observed_slices: set[QueryIntentSlice] = set()
        known_failure_ids: set[str] = set()
        for case in self.cases:
            if case.case_id in case_ids:
                raise ContractError(f"存在重复 case_id: {case.case_id}")
            case_ids.add(case.case_id)
            if (
                case.dataset_id != self.manifest.dataset_id
                or case.dataset_version != self.manifest.dataset_version
                or case.split != self.manifest.split
            ):
                raise ContractError(f"Case {case.case_id} 与 manifest 版本身份不一致")
            observed_slices.update(case.slices)
            if case.origin is QueryIntentCaseOrigin.DEVELOPMENT_KNOWN_FAILURE:
                known_failure_ids.add(case.case_id)

        expected_known_failures = set(self.manifest.known_failure_case_ids)
        if known_failure_ids != expected_known_failures:
            raise ContractError(
                "manifest 与 cases.jsonl 的 development_known_failure 清单不一致"
            )
        missing_slices = set(self.manifest.required_slices) - observed_slices
        if missing_slices:
            missing = ", ".join(sorted(item.value for item in missing_slices))
            raise ContractError(f"开发集缺少 required_slices: {missing}")


__all__ = [
    "QueryIntentCaseOrigin",
    "QueryIntentCategory",
    "QueryIntentDecision",
    "QueryIntentExpectedOutcome",
    "QueryIntentHistoryRole",
    "QueryIntentHistoryTurn",
    "QueryIntentReasonCode",
    "QueryIntentResponseKind",
    "QueryIntentSafetyCase",
    "QueryIntentSafetyDataset",
    "QueryIntentSafetyManifest",
    "QueryIntentSlice",
]
