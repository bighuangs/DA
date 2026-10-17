"""SQL 校正节点：基于校验证据生成受约束的修复策略，并安全更新候选 SQL。"""

import json
import logging

import yaml
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.agent_system.agents.nl2sql.context import DataAgentContext
from app.agent_system.agents.nl2sql.llm import get_task_llm, task_timeout
from app.agent_system.agents.nl2sql.state import DataAgentState, MAX_RETRY
from app.agent_system.agents.nl2sql.semantic_hints import build_semantic_contract
from app.agent_system.agents.nl2sql.semantic_contract import compact_query_contract
from app.agent_system.runtime.llm_attempts import normalize_degradation
from app.agent_system.agents.nl2sql.llm import (
    LLMInvocationError,
    invoke_llm,
    normalize_sql_output,
    pre_invoke_failure,
    sql_output_validator,
)
from app.agent_system.agents.nl2sql.services.prompt_context import compact_metric_infos, compact_table_infos
from app.agent_system.agents.nl2sql.llm import (
    build_model_clarification,
    clarification_message,
)
from app.agent_system.governance.deadline import model_phase_budget, route_deadline_at
from app.agent_system.agents.nl2sql.services.semantic_repair import (
    TARGETED_CORRECTION_PROFILES,
    UPSTREAM_FIRST_PROFILES,
    runtime_profile_status,
)
from app.domain.query.safety.semantic_violation import correction_directive, stable_hash
from app.agent_system.agents.nl2sql.prompts.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

# 校正任务使用独立配置的模型实例，避免与规划、生成等任务混用模型参数。
llm = None


def semantic_ir_for_correction_prompt(
    semantic_ir: dict | None,
) -> dict:
    """Expose parser details only when their shared contract is authoritative."""

    value = dict(semantic_ir or {})
    contract = compact_query_contract(value.get("query_contract"))
    if contract.get("authority") == "closed":
        return value
    return {
        "status": "guided_ir_withheld",
        "query_contract": contract,
        "model_sql_policy": dict(value.get("model_sql_policy") or {}),
        "withheld_reason": "model_route_parser_not_authoritative",
    }


def retain_valid_correction(previous_sql: str, candidate_sql: str | None) -> tuple[str, bool]:
    """
    在模型返回空结果时保留上一版可诊断 SQL。

    核心功能：将模型候选统一转换为去除首尾空白的字符串；候选有效时采用新 SQL，
    候选为空时回退到上一版 SQL，防止安全过滤或输出截断导致校验对象丢失。

    业务逻辑：
        1. 将空值候选归一化为空字符串并清理首尾空白
        2. 若候选非空，则返回新 SQL 并标记已接受
        3. 否则清理上一版 SQL 后返回，并标记未接受

    Args:
        previous_sql: 进入校正节点前、仍可用于诊断的 SQL。
        candidate_sql: 校正模型生成的新 SQL，可能为空或为 ``None``。

    Returns:
        tuple[str, bool]: 最终保留的 SQL，以及是否接受了模型候选。
    """
    # 统一处理 None、空串和纯空白输出，使接受条件只依赖清理后的实际内容。
    candidate = str(candidate_sql or "").strip()

    # 非空候选可进入下一轮确定性校验，并明确标记本次模型输出已被采用。
    if candidate:
        return candidate, True

    # 空候选不能覆盖已有 SQL；返回上一版内容，供后续校验继续提供诊断证据。
    return str(previous_sql or "").strip(), False


def build_correction_policy(
    validation_evidence: dict | None,
    *,
    settings=None,
) -> dict:
    """
    根据版本化违规证据构建有边界、感知运行配置的纠错策略。

    核心功能：读取确定性校验产生的违规对象和当前语义修复运行配置，生成模型可执行的
    纠错指令，并明确是否启用定向修复及上游优先修复，防止纠错越过原查询边界。

    业务逻辑：
        1. 将空校验证据归一化为空字典并提取违规对象
        2. 获取当前语义修复运行配置，缺失时回退到 ``inline_v1``
        3. 将违规对象转换为基础纠错指令
        4. 判断当前配置是否支持定向修复和上游优先修复
        5. 将运行配置及能力标记合并到指令中
        6. 若未启用定向修复，则覆盖为通用最小纠错约束

    Args:
        validation_evidence: SQL 校验阶段生成的证据，可包含版本化违规信息。

    Returns:
        dict: 包含纠错指令、运行配置及修复能力标记的策略。
    """
    # 空证据按空字典处理，确保缺少扩展违规信息时仍能生成通用修复策略。
    evidence = dict(validation_evidence or {})
    violation = dict(evidence.get("violation") or {})

    # 运行配置决定可启用的修复能力；未声明配置时使用内联基础版本。
    runtime_status = runtime_profile_status(settings)
    profile = str(runtime_status.get("runtime_profile") or "inline_v1")

    # 将结构化违规信息转换为给模型使用的、范围受控的基础纠错指令。
    directive = correction_directive(violation)
    targeted = profile in TARGETED_CORRECTION_PROFILES

    # 把运行配置及能力开关写入同一策略，供提示词和审计记录一致使用。
    directive.update({
        "profile": profile,
        "targeted": targeted,
        "upstream_first": profile in UPSTREAM_FIRST_PROFILES,
    })

    # 非定向配置仍保留已注册 violation 的最小结构指令。过去在这里
    # 覆盖 instruction 会让“缺少完整年度过滤”失去可执行修复目标，甚至
    # 把添加已经授权的日期维表误写成禁止动作。安全边界独立表达：只能在
    # 当前 table_infos 与元数据关系闭包内修复，绝不扩展授权范围。
    if not targeted:
        directive["boundary_instruction"] = (
            "只做该错误所需的最小修改；不得使用 table_infos 之外的表字段，"
            "但允许为恢复用户明确语义而添加已提供且元数据可证明 PK/FK 连接的表。"
            "不得改变指标、用户过滤、时间范围或权限边界。"
        )
    return directive


async def correct_sql(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    settings = runtime.context["settings"]
    """
    在 SQL 校验失败后执行一次受策略约束、可防重复且可回退的模型纠错。

    核心功能：将原始问题、数据上下文、上一版 SQL、语义契约、语义 IR、执行计划
    和版本化校验证据共同注入纠错提示词；调用专用模型生成最小修改后的 SQL，同时
    利用稳定哈希阻断相同 SQL 与错误码的重复修复，并在模型空输出时保留旧 SQL。

    业务逻辑：
        1. 向前端推送“校正SQL”阶段，并从状态中提取查询、元数据及上次错误 SQL
        2. 根据校验证据和运行配置构建有边界的纠错策略
        3. 计算 SQL 与违规码的稳定尝试标识；若已尝试过，则终止重复纠错循环
        4. 构建语义契约与提示词处理链，并序列化全部必要上下文
        5. 异步调用校正模型；若输出非空则采用，否则保留上一版 SQL
        6. 更新重试次数、修复尝试、授权状态及必要的降级证据
        7. 若任一阶段异常，则记录错误并向上抛出

    Args:
        state: 流水线状态，包含查询、数据上下文、校验证据、错误 SQL 和重试信息。
        runtime: LangGraph 运行时，提供节点上下文与前端流式写入器。

    Returns:
        dict: 状态增量；包含最终 SQL、重试次数、修复尝试及可选降级记录。

    Raises:
        Exception: 纠错策略构建、提示词调用或结果处理期间发生的任意异常。
    """
    # =============================================================================
    # 阶段 1：通知前端并提取纠错所需状态
    # =============================================================================

    # 在任何耗时操作前推送节点阶段，使调用方能够及时展示当前处理进度。
    runtime.stream_writer({"stage": "校正SQL"})
    try:
        # 依次提取生成纠错提示词所需的数据上下文、上一版 SQL 与校验错误；
        # 必需字段使用索引访问，让状态缺失问题能够立即暴露。
        query = state["query"]
        table_infos = state["table_infos"]
        metric_infos = state["metric_infos"]
        date_info = state["date_info"]
        db_info = state["db_info"]
        previous_sql = state["sql"]
        error = state["error"]
        plan = state.get("plan", "")
        validation_evidence = dict(state.get("validation_evidence") or {})
        violation = dict(validation_evidence.get("violation") or {})
        if violation and (
            violation.get("repairability") != "llm"
            or violation.get("safe_to_retry") is not True
        ):
            logger.warning(
                "拒绝越级模型纠错 violation_code=%s repairability=%s",
                violation.get("violation_code"),
                violation.get("repairability"),
            )
            return {
                "sql": previous_sql,
                "authorized_sql": None,
                "retry_count": MAX_RETRY,
                "error": error,
                "degradations": [{
                    "node_id": "correct_sql",
                    "reason_code": "semantic_repair.non_llm_terminal",
                    "message": "该违规不允许进入模型纠错",
                }],
            }

        # =============================================================================
        # 阶段 2：构建纠错策略与稳定尝试标识
        # =============================================================================

        # 校验证据与当前运行配置共同决定本次允许采用的纠错范围和能力。
        correction_policy = build_correction_policy(
            validation_evidence,
            settings=settings,
        )

        # 优先使用结构化违规码，其次使用通用错误码；两者都缺失时采用稳定默认值。
        violation_code = str(
            (validation_evidence.get("violation") or {}).get("violation_code")
            or validation_evidence.get("error_code")
            or "sql.validation_error"
        )

        # 先对 SQL 脱敏哈希，再与违规码组合生成尝试键，避免记录 SQL 明文。
        attempt_key = stable_hash(f"{stable_hash(previous_sql)}:{violation_code}")

        # 从历史修复记录中过滤字典项，忽略旧版本或异常格式写入的无效数据。
        previous_attempts = [
            item for item in (state.get("repair_attempts") or [])
            if isinstance(item, dict)
        ]

        # =============================================================================
        # 阶段 3：阻断相同 SQL 与违规码的重复修复
        # =============================================================================

        # 若相同尝试键已经出现，说明模型纠错未改变问题；立即耗尽重试预算，
        # 防止状态图在同一个语义错误上无限循环。
        if any(item.get("attempt_key") == attempt_key for item in previous_attempts):
            logger.warning(
                "跳过重复语义纠错 violation_code=%s sql_hash=%s",
                violation_code,
                stable_hash(previous_sql),
            )
            return {
                "sql": previous_sql,
                "authorized_sql": None,
                "retry_count": MAX_RETRY,
                "error": error,

                # 将重复修复阻断记录为降级证据，便于前端和遥测解释提前终止原因。
                "degradations": [{
                    "node": "correct_sql",
                    "reason": "duplicate_semantic_repair_attempt",
                    "message": "相同 SQL 与错误码已尝试纠错，已阻止重复循环",
                }],
            }

        # =============================================================================
        # 阶段 4：构建语义契约与模型调用链
        # =============================================================================

        timeout = task_timeout(runtime, "correction")
        phase_budget = model_phase_budget(
            state,
            "correction",
            minimum_start_ms=int(
                settings.query.correction_llm_min_start_budget_ms
            ),
            timeout_seconds=timeout,
        )
        if not phase_budget["allowed"]:
            clarification = build_model_clarification(
                node_id="correct_sql",
                reason_code=phase_budget["reason_code"],
                semantic_ir=state.get("semantic_ir") or {},
                retryable=True,
            )
            message = clarification_message(clarification)
            degradation = normalize_degradation({
                "node_id": "correct_sql",
                "route": "correction",
                "reason_code": "correction.skipped_insufficient_budget",
                "severity": "error",
                "affects_core_path": True,
                "message": "纠错预算不足，保留原 SQL 且禁止执行",
                "duration_ms": 0,
            })
            runtime.stream_writer({"clarification": clarification})
            runtime.stream_writer({"degradation": degradation})
            runtime.stream_writer({"error": message})
            return {
                "sql": previous_sql,
                "authorized_sql": None,
                "retry_count": MAX_RETRY,
                "error": message,
                "clarification": clarification,
                "model_phase_budget": phase_budget,
                "degradations": [degradation],
            }

        # 将当前问题、表元数据和对话历史压缩为必须遵守的语义契约，
        # 约束模型只修复已识别错误，不改变原始查询意图。
        semantic_contract = build_semantic_contract(
            query,
            table_infos,
            state.get("chat_history", []),
            metric_infos,
            state.get("semantic_ir") or {},
        )

        # 当前已重试次数会在本次校正结束后加一，供后续路由判断剩余机会。
        retry_count = state.get("retry_count", 0)

        # 加载 SQL 纠错模板并显式声明全部输入，确保模板缺参时尽早失败。
        prompt_template = PromptTemplate(
            template=load_prompt("correct_sql"),
            input_variables=["query", "table_infos", "metric_infos", "date_info", "db_info","sql","error","plan","semantic_contract","semantic_ir","repair_directive"],
        )

        # 模型输出按纯字符串解析，再通过管道将模板、专用模型和解析器串联。
        # Preserve the raw message envelope for typed truncation/empty-output
        # classification and provider evidence at the invocation boundary.
        chain = prompt_template | (llm or get_task_llm(runtime, "correction"))

        # =============================================================================
        # 阶段 5：序列化上下文并调用校正模型
        # =============================================================================

        # 表、指标、日期、数据库和语义 IR 使用可读 YAML 序列化；纠错策略使用紧凑
        # JSON 保持结构边界，连同旧 SQL 与错误信息一起驱动模型定向修复。
        try:
            payload = {
            "query": query,
            "table_infos": yaml.dump(table_infos,allow_unicode=True,sort_keys=False),
            "metric_infos": yaml.dump(metric_infos,allow_unicode=True,sort_keys=False),
            "date_info": yaml.dump(date_info,allow_unicode=True,sort_keys=False),
            "db_info": yaml.dump(db_info,allow_unicode=True,sort_keys=False),
            "sql": previous_sql,
            "error": error,
            "plan": plan,
            "semantic_contract": semantic_contract,
            "semantic_ir": yaml.dump(
                semantic_ir_for_correction_prompt(state.get("semantic_ir")),
                allow_unicode=True,
                sort_keys=False,
            ),
            "repair_directive": json.dumps(
                correction_policy, ensure_ascii=False, separators=(",", ":")
            ),
            }
            recovery_payload = {
                **payload,
                "table_infos": yaml.dump(
                    compact_table_infos(
                        table_infos,
                        description_max_chars=settings.query.prompt_description_max_chars,
                        example_limit=settings.query.prompt_column_example_limit,
                    ),
                    allow_unicode=True,
                    sort_keys=False,
                ),
                "metric_infos": yaml.dump(
                    compact_metric_infos(
                        metric_infos,
                        description_max_chars=settings.query.prompt_description_max_chars,
                    ),
                    allow_unicode=True,
                    sort_keys=False,
                ),
                "plan": "",
                # The directive and the previous SQL already identify the exact
                # repair surface; dropping the verbose IR on recovery reduces
                # transport risk without introducing any new evidence.
                "semantic_ir": "{}\n",
            }
            invocation = await invoke_llm(
                chain,
                payload,
                node_id="correct_sql",
                route="correction",
                timeout_seconds=timeout,
                max_retries=1,
                core_path=True,
                runtime=runtime,
                request_deadline_at=route_deadline_at(
                    state,
                    "correction",
                    timeout_seconds=timeout,
                ),
                recovery_payload=recovery_payload,
                recovery_min_ms=int(
                    settings.query.llm_recovery_min_budget_ms
                ),
                primary_min_ms=int(
                    settings.query.correction_llm_primary_min_budget_ms
                ),
                retryable_reason_codes={
                    "llm.connection_error",
                    "llm.rate_limited",
                    "llm.server_error",
                    "llm.stream_interrupted",
                    "llm.timeout",
                    "llm.empty_output",
                    "llm.truncated_output",
                    "llm.invalid_structured_output",
                },
                output_validator=sql_output_validator,
            )
            candidate_sql = normalize_sql_output(invocation.value)
        except LLMInvocationError as exc:
            logger.error(
                "校正SQL调用失败 code=%s error_type=%s",
                exc.reason_code,
                exc.error_type,
            )
            clarification = build_model_clarification(
                node_id="correct_sql",
                reason_code=exc.reason_code,
                semantic_ir=state.get("semantic_ir") or {},
                retryable=True,
            )
            message = clarification_message(clarification)
            runtime.stream_writer({"clarification": clarification})
            runtime.stream_writer({"error": message})
            return {
                "sql": previous_sql,
                "authorized_sql": None,
                "retry_count": MAX_RETRY,
                "error": message,
                "clarification": clarification,
                "model_phase_budget": phase_budget,
                "repair_attempts": [{
                    "attempt_key": attempt_key,
                    "violation_code": violation_code,
                    "profile": correction_policy["profile"],
                    "targeted": bool(correction_policy["targeted"]),
                    "sql_hash": stable_hash(previous_sql),
                    "outcome": exc.reason_code,
                }],
                "llm_attempts": exc.attempts,
                "degradations": exc.degradations,
            }

        # 模型输出非空时接受新 SQL；空输出则回退旧 SQL，避免覆盖可诊断状态。
        sql, accepted = retain_valid_correction(previous_sql, candidate_sql)

        # 分别记录候选是否真正被采用，便于区分正常修复与安全回退。
        if accepted:
            logger.info("校正SQL完成")
        else:
            logger.warning("校正SQL返回空结果，保留上一版 SQL 继续校验")

        # =============================================================================
        # 阶段 6：组装状态增量与修复审计证据
        # =============================================================================

        # 重试计数加一并清空旧 error，避免下一轮校验或执行读取已失效的错误；
        # authorized_sql 重置为空，强制修正后的 SQL 重新通过授权与确定性校验。
        return {
            "sql": sql,
            "authorized_sql": None,
            "retry_count": retry_count + 1,
            "error": None,

            # 记录稳定尝试键、违规码、配置和 SQL 哈希，以支持下一轮重复检测，
            # 同时避免在审计状态中复制 SQL 明文。
            "repair_attempts": [{
                "attempt_key": attempt_key,
                "violation_code": violation_code,
                "profile": correction_policy["profile"],
                "targeted": bool(correction_policy["targeted"]),
                "sql_hash": stable_hash(previous_sql),
            }],
            "llm_attempts": invocation.attempts,
            "model_phase_budget": phase_budget,

            # 仅在模型空输出时附加降级记录；正常接受候选时展开为空字典，
            # 从而不向状态中写入无意义的 degradations 字段。
            **({
                "degradations": [{
                    "node": "correct_sql",
                    "reason": "empty_model_output",
                    "message": "SQL 校正模型返回空结果，已保留上一版 SQL",
                }]
            } if not accepted else {}),
        }
    except Exception as exc:
        failure = pre_invoke_failure(
            node_id="correct_sql",
            route="correction",
            timeout_seconds=task_timeout(runtime, "correction"),
            error=exc,
            core_path=True,
            runtime=runtime,
        )
        logger.error(
            "校正SQL准备失败 code=%s error_type=%s",
            failure.reason_code,
            failure.error_type,
        )
        runtime.stream_writer({"error": failure.user_message})
        return {
            "sql": str(state.get("sql") or ""),
            "authorized_sql": None,
            "retry_count": MAX_RETRY,
            "error": failure.user_message,
            "llm_attempts": failure.attempts,
            "degradations": failure.degradations,
        }
