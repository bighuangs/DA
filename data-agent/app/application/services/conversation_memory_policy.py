"""分层会话记忆的应用策略与隐私安全 Prompt 上下文组装。"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

import sqlglot
from sqlglot import exp

from app.application.settings.conversation import (
    DEFAULT_MEMORY_SETTINGS,
    ConversationMemorySettings,
)
from app.domain.conversation.memory_models import MemoryContext
from app.domain.privacy.masking import (
    mask_free_text as _mask_free_text,
    mask_sql_literals as _mask_sql_literals,
)


MEMORY_ID_NAMESPACE = uuid.UUID("1710a430-c7c3-5c44-9ce7-31a20a456b40")
SUMMARY_ID_NAMESPACE = uuid.UUID("2f50a638-9e56-5aef-a719-9df26986726d")
SUMMARY_SIGNAL_ID_NAMESPACE = uuid.UUID("9c01b0f6-6bdf-56cb-b015-39de95002ca2")
_PREFERENCE_RE = re.compile(
    r"(?:以后|今后|默认|始终|每次|请一直|回答用|输出用|优先用|不要再)"
)
_ASCII_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+")
_CHINESE_CHUNK_RE = re.compile(r"[\u4e00-\u9fff]{2,}")
_LONG_RANGE_MARKERS = (
    "之前", "最开始", "前面", "当时", "早先", "上次", "继续", "那个",
    "原来的", "先前", "还记得", "长期目标", "我说过",
)
_SUMMARY_SIGNAL_STOPWORDS = {
    "用户", "问题", "查询", "成功", "返回", "多少", "哪些", "什么", "如何",
    "每个", "各个", "以及", "进行", "这个", "那个", "之前", "当时", "继续",
}
_SENSITIVE_IDENTITY_RE = re.compile(
    r"(?:"
    r"宗教(?:信仰)?|信教|信佛|信仰(?:佛教|基督教|天主教|伊斯兰教|道教)|"
    r"佛教徒?|基督(?:教|徒)|天主教徒?|伊斯兰教|穆斯林|"
    r"道教徒?|犹太教徒?|犹太人|锡克教徒?|印度教徒?|无神论者?|"
    r"政治(?:面貌|身份|立场)?|党派|党员|共产党员?|共青团员|"
    r"民主党|共和党|民盟|民进|民建|农工党|致公党|九三学社|"
    r"工会会员|左翼|右翼|自由派|保守派|马克思主义者|"
    r"种族|族裔|民族|少数民族|汉族|回族|回民|满族|蒙古族|"
    r"藏族|藏人|维吾尔族|维族|壮族|苗族|彝族|华裔|亚裔|拉丁裔|黑人|白人|国籍|"
    r"性取向|同性恋|异性恋|双性恋|无性恋|同性伴侣|"
    r"喜欢同性|lgbtq?\+?|gay|lesbian|queer|性别认同|跨性别|非二元性别|"
    r"残疾|残障|身体障碍|肢体障碍|视力障碍|听力障碍|智力障碍|精神障碍|盲人|聋人|"
    r"失明|失聪|坐轮椅|行动不便|"
    r"病史|手术史|健康状况|疾病|患(?:有|过)|得过|确诊|患者|病人|过敏|"
    r"高血压|糖尿病|癌症|肿瘤|抑郁症?|焦虑症?|精神分裂|"
    r"自闭症|唐氏综合征|艾滋病|hiv|乙肝|哮喘|"
    r"精确住址|未成年人|身份证|银行卡|生物特征|指纹|人脸"
    r")",
    re.IGNORECASE,
)
_PROFILE_ROLE_RE = re.compile(
    r"(?:我是|我的(?:岗位|职位|职业|工作|角色)是)"
    r"([^，。！？!?]{2,60})"
)
_PROFILE_RESPONSIBILITY_RE = re.compile(
    r"(?:我负责|我的(?:工作职责|岗位职责|主要职责|职责)(?:是|包括)?)"
    r"([^，。！？!?]{2,80})"
)
_WORK_ROLE_ALLOWLIST_RE = re.compile(
    r"^(?:(?:一名|一位|一个)\s*)?"
    r"(?:(?:资深|高级|中级|初级|首席|助理)\s*)?"
    r"(?:"
    r"(?:(?:数据|商业|业务|销售|客户|市场|运营|产品|项目|"
    r"软件|后端|前端|全栈|算法|机器学习|人工智能|安全|云|数据库|"
    r"网络|系统|财务|审计|法务|采购|供应链|人力资源)\s*){0,2}"
    r"(?:分析师|工程师|程序员|开发者|架构师|设计师|科学家|"
    r"经理|主管|总监|专员|顾问|负责人|管理员|研究员|"
    r"研发|测试|运维|运营|销售|客服|财务|会计|审计|法务|"
    r"律师|采购|行政|人力资源|教师|医生|护士|编辑|记者|翻译)"
    r"|(?:(?:senior|principal|lead|junior)\s+)?(?:data\s+analyst|"
    r"business\s+analyst|software\s+engineer|data\s+engineer|"
    r"product\s+manager|project\s+manager|developer|programmer|designer|"
    r"researcher|consultant|accountant|auditor|sales|operations?|"
    r"ceo|cto|cfo|coo|hr)"
    r")(?:岗位|职位|人员)?$",
    re.IGNORECASE,
)
_WORK_RESPONSIBILITY_ALLOWLIST_RE = re.compile(
    r"^(?:(?:华东|华南|华北|华中|东北|西北|西南|全国|海外)"
    r"(?:区|区域)?|数据|业务|销售|客户|订单|收入|利润|指标|"
    r"报表|项目|产品|研发|开发|测试|运维|运营|市场|营销|"
    r"财务|审计|法务|采购|库存|供应链|风控|系统|代码|数仓|"
    r"团队|部门|人力资源|办公室|分析|管理|维护|设计|建设|"
    r"优化|监控|支持|规划|整理|审核|服务|与|和|及|、|/|-|\s|"
    r"sql|analytics?|engineering|operations?|sales|finance|product|project)+$",
    re.IGNORECASE,
)
_WORK_IDENTITY_STATEMENT_RE = re.compile(
    r"^(?:我|本人|用户|该用户|你|您|他|她)"
    r"(?:是|的(?:岗位|职位|职业|工作|角色)是)"
    r"(.+)$"
)
_PERSONAL_IDENTITY_ASSERTION_RE = re.compile(
    r"(?:我|本人|用户|该用户|你|您|他|她|其)"
    r"(?:的(?:身份|民族|族裔|种族|国籍|信仰|宗教|政治面貌|"
    r"党派|性取向|性别认同|健康状况|病史|残疾状况)是|"
    r"是|为|属于|来自|加入|信奉|信仰)"
)
_GENERIC_SENSITIVE_IDENTITY_SHAPE_RE = re.compile(
    r"(?:[一-鿿]{1,8}(?:族|裔)(?:人|人士|成员|后裔)?|"
    r"[一-鿿]{1,8}(?:教徒|信徒|信众)|"
    r"[一-鿿]{1,10}(?:党员|党派成员|盟员|团员)|"
    r"(?:患有|患上|患过|得过|确诊为).{0,24}(?:病|症|障碍)|"
    r"(?:我|本人|用户|该用户|你|您|他|她)有.{0,24}"
    r"(?:病|症|障碍|残疾|残障|过敏))",
    re.IGNORECASE,
)
_SENSITIVE_SUMMARY_PLACEHOLDER = "敏感身份内容已省略"
_LONG_TERM_GOAL_RE = re.compile(
    r"(?:我的长期目标是|长期目标是|我的目标是)([^，。！？!?]{2,80})"
)


def estimate_tokens(value: Any) -> int:
    """保守估算中英文混合 JSON 的 token 数，供记忆预算裁剪使用。"""

    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    chinese = len(re.findall(r"[\u4e00-\u9fff]", raw))
    return max(1, chinese + (len(raw) - chinese + 3) // 4)


def preference_slot(text: str) -> str:
    """Map explicit preferences to a small, explainable supersession domain."""
    value = str(text or "").lower()
    rules = (
        ("language", ("中文", "英文", "语言", "回答用", "用汉语")),
        ("ordering", ("排序", "升序", "降序", "从高到低", "从低到高")),
        (
            "presentation",
            ("表格", "图表", "折线", "柱状", "饼图", "展示方式", "呈现方式"),
        ),
        ("number_format", ("小数", "百分比", "千分位", "万元", "亿元")),
        ("verbosity", ("简洁", "详细", "只给", "解释过程", "不要解释")),
    )
    for slot, markers in rules:
        if any(marker in value for marker in markers):
            return slot
    # Unknown preferences stay independent instead of revoking unrelated facts.
    return "independent"


def _lexical_tokens(text: str) -> set[str]:
    """Build deterministic Chinese fallback tokens for MySQL-only retrieval."""
    value = str(text or "").lower()
    tokens = {item.lower() for item in _ASCII_TOKEN_RE.findall(value)}
    for chunk in _CHINESE_CHUNK_RE.findall(value):
        tokens.add(chunk)
        for width in range(2, min(4, len(chunk)) + 1):
            tokens.update(
                chunk[index:index + width]
                for index in range(0, len(chunk) - width + 1)
            )
    return tokens


def sql_fingerprint(
    sql: str | None,
    *,
    settings: ConversationMemorySettings = DEFAULT_MEMORY_SETTINGS,
) -> str | None:
    if not sql:
        return None
    try:
        normalized = sqlglot.parse_one(str(sql), read="mysql").sql(
            dialect="mysql", pretty=False, normalize=True
        )
    except Exception:
        normalized = " ".join(
            _mask_sql_literals(str(sql), policy=settings.masking_policy).split()
        )
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def sql_table_ids(sql: str | None) -> list[str]:
    if not sql:
        return []
    try:
        tree = sqlglot.parse_one(str(sql), read="mysql")
    except Exception:
        return []
    return sorted({
        str(table.name or "").strip()
        for table in tree.find_all(exp.Table)
        if str(table.name or "").strip()
    })


def deterministic_memory_id(
    *,
    owner_id: str,
    workspace_id: str,
    memory_type: str,
    source_message_ids: Sequence[int],
    extractor_version: str,
) -> str:
    identity = "|".join((
        owner_id,
        workspace_id,
        memory_type,
        ",".join(str(value) for value in source_message_ids),
        extractor_version,
    ))
    return str(uuid.uuid5(MEMORY_ID_NAMESPACE, identity))


def _value(row: object, name: str, default=None):
    if isinstance(row, Mapping):
        return row.get(name, default)
    return getattr(row, name, default)


def _is_allowlisted_work_role(value: str) -> bool:
    normalized = str(value or "").strip()
    return _WORK_ROLE_ALLOWLIST_RE.fullmatch(normalized) is not None


def _is_allowlisted_work_responsibility(value: str) -> bool:
    normalized = str(value or "").strip()
    return _WORK_RESPONSIBILITY_ALLOWLIST_RE.fullmatch(normalized) is not None


def _is_allowlisted_work_statement(text: str) -> bool:
    normalized = str(text or "").strip()
    normalized = re.sub(
        r"^(?:用户曾询问或陈述|用户曾询问|用户曾陈述)[:：]\s*",
        "",
        normalized,
    )
    role_match = _WORK_IDENTITY_STATEMENT_RE.fullmatch(normalized)
    if role_match and _is_allowlisted_work_role(role_match.group(1)):
        return True
    responsibility_match = _PROFILE_RESPONSIBILITY_RE.fullmatch(normalized)
    return bool(
        responsibility_match
        and _is_allowlisted_work_responsibility(responsibility_match.group(1))
    )


def _contains_sensitive_identity(text: str) -> bool:
    normalized = str(text or "").strip()
    if not normalized:
        return False
    if (
        _SENSITIVE_IDENTITY_RE.search(normalized)
        or _GENERIC_SENSITIVE_IDENTITY_SHAPE_RE.search(normalized)
    ):
        return True
    clauses = re.split(r"[，,。；;！!？?：:\n]+", normalized)
    return any(
        _PERSONAL_IDENTITY_ASSERTION_RE.search(clause)
        and not _is_allowlisted_work_statement(clause)
        for clause in clauses
        if clause.strip()
    )


def _sanitize_summary_source(
    text: str,
    *,
    max_chars: int,
    settings: ConversationMemorySettings = DEFAULT_MEMORY_SETTINGS,
) -> tuple[str, bool]:
    """Remove the whole source line when it may expose a sensitive identity."""
    masked = _mask_free_text(
        str(text or ""),
        policy=settings.masking_policy,
        max_chars=max_chars,
    )
    if not masked:
        return "", False
    safe_lines: list[str] = []
    redacted = False
    for line in masked.splitlines() or [masked]:
        normalized = str(line or "").strip()
        if not normalized:
            continue
        if _contains_sensitive_identity(normalized):
            safe_lines.append(_SENSITIVE_SUMMARY_PLACEHOLDER)
            redacted = True
        else:
            safe_lines.append(normalized)
    return "\n".join(safe_lines), redacted


def _safe_summary_fact(
    fact: object,
    *,
    settings: ConversationMemorySettings = DEFAULT_MEMORY_SETTINGS,
) -> dict | None:
    """Defensively sanitize facts copied from a lower summary level."""
    if not isinstance(fact, Mapping):
        return None
    row = dict(fact)
    source_ids = [
        int(value)
        for value in list(row.get("source_message_ids") or [])
        if int(value) > 0
    ]
    serialized = json.dumps(row, ensure_ascii=False, default=str)
    content, redacted = _sanitize_summary_source(
        str(row.get("content") or ""),
        max_chars=int(settings.message_content_max_chars),
        settings=settings,
    )
    redacted = redacted or bool(
        _SENSITIVE_IDENTITY_RE.search(serialized)
        or _GENERIC_SENSITIVE_IDENTITY_SHAPE_RE.search(serialized)
    )
    if redacted:
        return {
            "fact_type": "redacted_sensitive_identity",
            "content": _SENSITIVE_SUMMARY_PLACEHOLDER,
            "source_message_ids": source_ids,
            "status": "redacted",
        }
    if not content:
        return None
    return {
        "fact_type": str(row.get("fact_type") or "user_statement"),
        "content": content,
        "source_message_ids": source_ids,
        "status": str(row.get("status") or "supported"),
    }


def summary_content_hash(
    messages: Sequence[object],
    *,
    structured_facts: Sequence[Mapping[str, Any]] = (),
    schema_version: str | None = None,
    settings: ConversationMemorySettings = DEFAULT_MEMORY_SETTINGS,
) -> str:
    """Hash every authoritative input to a summary identity.

    The digest includes the masked transcript, sanitized semantic facts, and
    schema version.  A change in any of those inputs therefore cannot silently
    reuse an older segment identifier.
    """
    normalized = [
        {
            "id": int(_value(row, "id", _value(row, "message_id", 0)) or 0),
            "role": str(_value(row, "role", "")),
            "status": str(_value(row, "status", "success")),
            "content": " ".join(
                str(_value(row, "content_masked", _value(row, "content", "")) or "")
                .strip()
                .split()
            ),
        }
        for row in messages
    ]
    safe_facts = [
        safe
        for item in structured_facts
        if (safe := _safe_summary_fact(item, settings=settings)) is not None
    ]
    raw = json.dumps(
        {
            "messages": normalized,
            "structured_facts": safe_facts,
            "schema_version": str(
                schema_version or settings.summary_schema_version
            ),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def deterministic_summary_id(
    *,
    owner_id: str,
    workspace_id: str,
    chat_id: str,
    level: int,
    source_hash: str,
    extractor_version: str,
) -> str:
    identity = "|".join((
        str(owner_id), str(workspace_id), str(chat_id), str(int(level)),
        str(source_hash), str(extractor_version),
    ))
    return str(uuid.uuid5(SUMMARY_ID_NAMESPACE, identity))


def _summary_signal_type(value: str) -> str:
    text = str(value or "")
    if re.search(r"20\d{2}|季度|月份|今年|去年|本月|上月|同比|环比", text):
        return "time"
    if any(marker in text for marker in (
        "销售", "订单", "金额", "收入", "利润", "客户", "数量", "复购", "客单", "占比",
    )):
        return "metric"
    if "目标" in text:
        return "goal"
    if any(marker in text for marker in ("默认", "偏好", "始终", "优先")):
        return "preference"
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", text):
        return "identifier"
    return "topic"


def _summary_signal_candidates(
    content: str,
    *,
    settings: ConversationMemorySettings = DEFAULT_MEMORY_SETTINGS,
) -> list[str]:
    safe_content, _ = _sanitize_summary_source(
        content,
        max_chars=int(settings.summary_content_max_chars),
        settings=settings,
    )
    masked = "\n".join(
        line
        for line in safe_content.lower().splitlines()
        if _SENSITIVE_SUMMARY_PLACEHOLDER not in line
    )
    if not masked:
        return []
    candidates: list[str] = []
    candidates.extend(token.lower() for token in _ASCII_TOKEN_RE.findall(masked))
    for chunk in _CHINESE_CHUNK_RE.findall(masked):
        if 2 <= len(chunk) <= 8:
            candidates.append(chunk)
        for width in (4, 3, 2):
            if len(chunk) < width:
                continue
            candidates.extend(
                chunk[index:index + width]
                for index in range(0, len(chunk) - width + 1)
            )
    unique: list[str] = []
    for value in candidates:
        normalized = " ".join(str(value).strip().split())
        if (
            len(normalized) < 2
            or normalized in _SUMMARY_SIGNAL_STOPWORDS
            or (normalized.isdigit() and not re.fullmatch(r"20\d{2}", normalized))
            or normalized in unique
        ):
            continue
        unique.append(normalized)
    return unique


def _ordered_summary_signal_values(
    values: Iterable[str],
    *,
    limit: int | None,
    settings: ConversationMemorySettings = DEFAULT_MEMORY_SETTINGS,
) -> list[str]:
    maximum = max(1, int(limit or settings.summary_signal_limit))
    return sorted(
        set(values),
        key=lambda value: (
            0 if _summary_signal_type(value) in {"metric", "time", "goal", "preference"} else 1,
            -min(len(value), 8),
            value,
        ),
    )[:maximum]


def _extract_summary_signals_from_sources(
    sources: Sequence[tuple[str, Sequence[int]]],
    *,
    limit: int | None = None,
    settings: ConversationMemorySettings = DEFAULT_MEMORY_SETTINGS,
) -> list[dict]:
    """Extract signals while retaining only messages that contain each signal."""
    provenance: dict[str, set[int]] = {}
    for content, source_message_ids in sources:
        source_ids = {
            int(value)
            for value in source_message_ids
            if int(value) > 0
        }
        for value in _summary_signal_candidates(content, settings=settings):
            provenance.setdefault(value, set()).update(source_ids)
    ordered = _ordered_summary_signal_values(
        provenance,
        limit=limit,
        settings=settings,
    )
    return [
        {
            "signal_type": _summary_signal_type(value),
            "normalized_value": value,
            "weight": round(max(0.4, 1.0 - index * 0.04), 4),
            "source_order": index,
            "source_message_ids": sorted(provenance[value]),
        }
        for index, value in enumerate(ordered)
    ]


def extract_summary_signals(
    content: str,
    *,
    source_message_ids: Sequence[int],
    limit: int | None = None,
    settings: ConversationMemorySettings = DEFAULT_MEMORY_SETTINGS,
) -> list[dict]:
    """Build an independent, deterministic SummarySignalsV1 record set."""
    return _extract_summary_signals_from_sources(
        [(content, source_message_ids)],
        limit=limit,
        settings=settings,
    )


def deterministic_summary_signal_id(
    *, summary_id: str, signal_type: str, normalized_value: str, extractor_version: str
) -> str:
    identity = "|".join((
        str(summary_id), str(signal_type), str(normalized_value), str(extractor_version)
    ))
    return str(uuid.uuid5(SUMMARY_SIGNAL_ID_NAMESPACE, identity))


def build_extractive_summary_segment(
    *,
    owner_id: str,
    workspace_id: str,
    chat_id: str,
    messages: Sequence[object],
    structured_facts: Sequence[Mapping[str, Any]] = (),
    level: int = 0,
    child_summary_ids: Sequence[str] = (),
    settings: ConversationMemorySettings = DEFAULT_MEMORY_SETTINGS,
) -> tuple[dict, list[dict]]:
    """Create a source-grounded segment without an LLM or executable SQL."""
    if not messages:
        raise ValueError("summary segment 缺少 source messages")
    source_ids = [
        int(_value(row, "id", _value(row, "message_id", 0)) or 0)
        for row in messages
    ]
    if not source_ids or any(value <= 0 for value in source_ids):
        raise ValueError("summary segment source message id 无效")
    lines: list[str] = []
    facts: list[dict] = []
    signal_sources: list[tuple[str, Sequence[int]]] = []
    for row, message_id in zip(messages, source_ids):
        role = str(_value(row, "role", ""))
        content, sensitive_redacted = _sanitize_summary_source(
            str(_value(row, "content_masked", _value(row, "content", "")) or ""),
            max_chars=int(settings.message_content_max_chars),
            settings=settings,
        )
        if not content:
            continue
        if sensitive_redacted:
            lines.append(f"消息 {message_id}：{_SENSITIVE_SUMMARY_PLACEHOLDER}。")
            if role == "user":
                facts.append({
                    "fact_type": "redacted_sensitive_identity",
                    "content": _SENSITIVE_SUMMARY_PLACEHOLDER,
                    "source_message_ids": [message_id],
                    "status": "redacted",
                })
            continue
        if role == "user":
            lines.append(f"用户曾询问或陈述：{content}")
            facts.append({
                "fact_type": "user_statement",
                "content": content,
                "source_message_ids": [message_id],
                "status": "supported",
            })
            signal_sources.append((content, [message_id]))
        elif not content.startswith("查询成功，返回"):
            lines.append(f"助手已交付：{content}")
            signal_sources.append((content, [message_id]))
    maximum = int(settings.summary_content_max_chars)
    content_masked = _mask_free_text(
        "\n".join(lines),
        policy=settings.masking_policy,
        max_chars=maximum,
    )
    if not content_masked:
        content_masked = "该历史分段仅包含已完成查询的受控状态信息。"
    facts.extend(
        safe
        for item in structured_facts
        if (safe := _safe_summary_fact(item, settings=settings)) is not None
    )
    source_hash = summary_content_hash(
        messages,
        structured_facts=facts,
        schema_version=str(settings.summary_schema_version),
        settings=settings,
    )
    extractor_version = str(settings.summary_extractor_version)
    summary_id = deterministic_summary_id(
        owner_id=owner_id,
        workspace_id=workspace_id,
        chat_id=chat_id,
        level=level,
        source_hash=source_hash,
        extractor_version=extractor_version,
    )
    segment = {
        "summary_id": summary_id,
        "owner_id": owner_id,
        "workspace_id": workspace_id,
        "chat_id": chat_id,
        "level": int(level),
        "source_start_message_id": min(source_ids),
        "source_end_message_id": max(source_ids),
        "source_message_ids": source_ids,
        "child_summary_ids": [str(value) for value in child_summary_ids],
        "source_content_hash": source_hash,
        "content_masked": content_masked,
        "structured_facts": facts,
        "open_items": [],
        "source_token_count": estimate_tokens([
            str(_value(row, "content_masked", _value(row, "content", "")) or "")
            for row in messages
        ]),
        "summary_token_count": estimate_tokens(content_masked),
        "summary_schema_version": str(settings.summary_schema_version),
        "extractor_version": extractor_version,
        "model_version": None,
        "prompt_version": None,
        "projection_version": (
            f"es:{settings.summary_es_projection_version}|"
            f"qdrant:{settings.summary_qdrant_projection_version}"
        ),
        "status": "active",
    }
    signal_version = str(settings.summary_signal_extractor_version)
    signals = []
    for signal in _extract_summary_signals_from_sources(
        signal_sources,
        settings=settings,
    ):
        signals.append({
            **signal,
            "signal_id": deterministic_summary_signal_id(
                summary_id=summary_id,
                signal_type=signal["signal_type"],
                normalized_value=signal["normalized_value"],
                extractor_version=signal_version,
            ),
            "summary_id": summary_id,
            "owner_id": owner_id,
            "workspace_id": workspace_id,
            "chat_id": chat_id,
            "extractor_version": signal_version,
            "sensitivity": "masked",
            "status": "active",
        })
    return segment, signals


def build_hierarchical_summary_segment(
    *,
    owner_id: str,
    workspace_id: str,
    chat_id: str,
    children: Sequence[object],
    settings: ConversationMemorySettings = DEFAULT_MEMORY_SETTINGS,
) -> tuple[dict, list[dict]]:
    """Merge contiguous child segments while preserving their exact provenance."""
    if len(children) < 2:
        raise ValueError("hierarchical summary 至少需要两个 child segments")
    levels = {int(_value(row, "level", 0) or 0) for row in children}
    if len(levels) != 1:
        raise ValueError("hierarchical summary children 必须处于同一层级")
    level = next(iter(levels)) + 1
    source_ids = sorted({
        int(message_id)
        for row in children
        for message_id in list(_value(row, "source_message_ids", []) or [])
    })
    child_ids = [str(_value(row, "summary_id", "")) for row in children]
    child_contents: list[str] = []
    facts: list[dict] = []
    signal_sources: list[tuple[str, Sequence[int]]] = []
    for row in children:
        child_source_ids = [
            int(value)
            for value in list(_value(row, "source_message_ids", []) or [])
        ]
        child_content, _ = _sanitize_summary_source(
            str(_value(row, "content_masked", "") or ""),
            max_chars=int(settings.summary_content_max_chars),
            settings=settings,
        )
        if child_content:
            child_contents.append(child_content)
        child_facts = [
            safe_fact
            for fact in list(_value(row, "structured_facts", []) or [])
            if (safe_fact := _safe_summary_fact(fact, settings=settings)) is not None
        ]
        facts.extend(child_facts)
        safe_signal_facts = [
            fact
            for fact in child_facts
            if fact.get("status") != "redacted"
        ]
        if safe_signal_facts:
            for fact in safe_signal_facts:
                signal_sources.append((
                    str(fact.get("content") or ""),
                    list(fact.get("source_message_ids") or child_source_ids),
                ))
        elif child_content:
            signal_sources.append((child_content, child_source_ids))
    content = _mask_free_text(
        "\n".join(child_contents),
        policy=settings.masking_policy,
        max_chars=int(settings.summary_content_max_chars),
    )
    source_hash = "sha256:" + hashlib.sha256(
        "|".join(
            f"{child_id}:{_value(row, 'source_content_hash', '')}"
            for child_id, row in zip(child_ids, children)
        ).encode("utf-8")
    ).hexdigest()
    extractor_version = str(settings.summary_extractor_version)
    summary_id = deterministic_summary_id(
        owner_id=owner_id,
        workspace_id=workspace_id,
        chat_id=chat_id,
        level=level,
        source_hash=source_hash,
        extractor_version=extractor_version,
    )
    segment = {
        "summary_id": summary_id,
        "owner_id": owner_id,
        "workspace_id": workspace_id,
        "chat_id": chat_id,
        "level": level,
        "source_start_message_id": min(source_ids),
        "source_end_message_id": max(source_ids),
        "source_message_ids": source_ids,
        "child_summary_ids": child_ids,
        "source_content_hash": source_hash,
        "content_masked": content,
        "structured_facts": facts,
        "open_items": [
            item
            for row in children
            for item in list(_value(row, "open_items", []) or [])
        ],
        "source_token_count": sum(
            int(_value(row, "source_token_count", 0) or 0) for row in children
        ),
        "summary_token_count": estimate_tokens(content),
        "summary_schema_version": str(settings.summary_schema_version),
        "extractor_version": extractor_version,
        "model_version": None,
        "prompt_version": None,
        "projection_version": (
            f"es:{settings.summary_es_projection_version}|"
            f"qdrant:{settings.summary_qdrant_projection_version}"
        ),
        "status": "active",
    }
    signal_version = str(settings.summary_signal_extractor_version)
    signals = []
    for signal in _extract_summary_signals_from_sources(
        signal_sources,
        settings=settings,
    ):
        signals.append({
            **signal,
            "signal_id": deterministic_summary_signal_id(
                summary_id=summary_id,
                signal_type=signal["signal_type"],
                normalized_value=signal["normalized_value"],
                extractor_version=signal_version,
            ),
            "summary_id": summary_id,
            "owner_id": owner_id,
            "workspace_id": workspace_id,
            "chat_id": chat_id,
            "extractor_version": signal_version,
            "sensitivity": "masked",
            "status": "active",
        })
    return segment, signals


def summary_retrieval_needed(
    query: str,
    *,
    has_active_summary: bool,
    omitted_recent: int = 0,
    force: bool = False,
) -> tuple[bool, str]:
    if not has_active_summary:
        return False, "no_summary"
    if force:
        return True, "profile_always_retrieve"
    normalized = str(query or "").strip().lower()
    if any(marker in normalized for marker in _LONG_RANGE_MARKERS):
        return True, "long_range_reference"
    if omitted_recent > 0 and any(marker in normalized for marker in ("它", "其", "同样", "再", "也")):
        return True, "recent_history_omitted"
    return False, "not_needed"


def extract_explicit_stable_facts(
    *,
    owner_id: str,
    workspace_id: str,
    chat_id: str,
    query: str,
    user_message_id: int,
    settings: ConversationMemorySettings = DEFAULT_MEMORY_SETTINGS,
) -> list[dict]:
    """Extract only explicit, non-sensitive profile facts and long-term goals."""
    raw = str(query or "").strip()
    if not raw or _contains_sensitive_identity(raw):
        return []
    extractor = str(settings.stable_fact_extractor_version)
    candidates: list[tuple[str, str, str]] = []
    role_match = _PROFILE_ROLE_RE.search(raw)
    role_value = str(role_match.group(1) if role_match else "").strip()
    if (
        role_value
        and _is_allowlisted_work_role(role_value)
        and not any(marker in role_value for marker in ("多少", "什么", "哪些", "吗"))
    ):
        candidates.append(("profile_fact", role_value, "profile_context"))

    responsibility_match = _PROFILE_RESPONSIBILITY_RE.search(raw)
    responsibility_value = str(
        responsibility_match.group(1) if responsibility_match else ""
    ).strip()
    if (
        responsibility_value
        and _is_allowlisted_work_responsibility(responsibility_value)
        and not any(
            marker in responsibility_value
            for marker in ("多少", "什么", "哪些", "吗")
        )
    ):
        candidates.append(("profile_fact", responsibility_value, "profile_context"))

    goal_match = _LONG_TERM_GOAL_RE.search(raw)
    goal_value = str(goal_match.group(1) if goal_match else "").strip()
    if (
        goal_value
        and not any(marker in goal_value for marker in ("多少", "什么", "哪些", "吗"))
    ):
        candidates.append(("long_term_goal", goal_value, "long_term_goal"))
    rows: list[dict] = []
    for memory_type, value, slot in candidates:
        content = _mask_free_text(
            value,
            policy=settings.masking_policy,
            max_chars=300,
        )
        rows.append({
            "memory_id": deterministic_memory_id(
                owner_id=owner_id,
                workspace_id=workspace_id,
                memory_type=memory_type,
                source_message_ids=[int(user_message_id)],
                extractor_version=extractor,
            ),
            "owner_id": owner_id,
            "workspace_id": workspace_id,
            "chat_id": chat_id,
            "memory_type": memory_type,
            "content_masked": content,
            "structured_payload": {
                "explicit": True,
                "source_role": "user",
                "fact_slot": slot,
                "untrusted_text": True,
            },
            "source_message_ids": [int(user_message_id)],
            "data_source_id": "conversation",
            "schema_version": None,
            "sql_fingerprint": None,
            "confidence": 1.0,
            "verification_status": "confirmed",
            "sensitivity": "masked",
            "extractor_version": extractor,
            "embedding_version": (
                f"{settings.embedding_model}@{settings.embedding_revision}"
            ),
            "projection_version": (
                f"es:{settings.es_projection_version}|"
                f"qdrant:{settings.qdrant_projection_version}"
            ),
            "status": "active",
        })
    return rows


def extract_memory_candidates(
    *,
    owner_id: str,
    workspace_id: str,
    chat_id: str,
    query: str,
    standalone_question: str,
    follow_up_action: str,
    semantic_state: Mapping | None,
    sql: str,
    user_message_id: int,
    assistant_message_id: int,
    schema_version: str | None,
    settings: ConversationMemorySettings = DEFAULT_MEMORY_SETTINGS,
) -> list[dict]:
    """Deterministically extract only verified, reusable memory facts.

    Raw rows and executable SQL never enter memory.  SQL contributes only a
    fingerprint and current table identifiers; every future query must recall
    current schema and pass authorization again.
    """
    extractor = settings.extractor_version
    source_ids = [int(user_message_id), int(assistant_message_id)]
    masked_question = _mask_free_text(
        standalone_question,
        policy=settings.masking_policy,
        max_chars=settings.memory_content_max_chars,
    )
    common = {
        "owner_id": owner_id,
        "workspace_id": workspace_id,
        "chat_id": chat_id,
        "source_message_ids": source_ids,
        "data_source_id": "dw-main",
        "schema_version": schema_version,
        "confidence": 1.0,
        "verification_status": "executed",
        "sensitivity": "masked",
        "extractor_version": extractor,
        "embedding_version": (
            f"{settings.embedding_model}@{settings.embedding_revision}"
        ),
        "projection_version": (
            f"es:{settings.es_projection_version}|"
            f"qdrant:{settings.qdrant_projection_version}"
        ),
        "status": "active",
    }
    episode = {
        **common,
        "memory_id": deterministic_memory_id(
            owner_id=owner_id,
            workspace_id=workspace_id,
            memory_type="query_episode",
            source_message_ids=source_ids,
            extractor_version=extractor,
        ),
        "memory_type": "query_episode",
        "content_masked": masked_question,
        "structured_payload": {
            "standalone_question": masked_question,
            "semantic_state": dict(semantic_state or {}),
            "table_ids": sql_table_ids(sql),
            "follow_up_action": str(follow_up_action or "new_topic")[:32],
            "sql_reusable": False,
            "untrusted_text": True,
        },
        "sql_fingerprint": sql_fingerprint(sql, settings=settings),
    }
    candidates = [episode]

    original_masked = _mask_free_text(
        query,
        policy=settings.masking_policy,
        max_chars=settings.memory_content_max_chars,
    )
    if _PREFERENCE_RE.search(str(query or "")):
        candidates.append({
            **common,
            "memory_id": deterministic_memory_id(
                owner_id=owner_id,
                workspace_id=workspace_id,
                memory_type="preference",
                source_message_ids=[user_message_id],
                extractor_version=extractor,
            ),
            "memory_type": "preference",
            "content_masked": original_masked,
            "structured_payload": {
                "explicit": True,
                "untrusted_text": True,
                "priority": "user_preference",
                "preference_slot": preference_slot(query),
            },
            "sql_fingerprint": None,
            "verification_status": "confirmed",
        })

    if str(follow_up_action or "") in {"replace", "remove", "reset", "append"}:
        candidates.append({
            **common,
            "memory_id": deterministic_memory_id(
                owner_id=owner_id,
                workspace_id=workspace_id,
                memory_type="correction",
                source_message_ids=[user_message_id],
                extractor_version=extractor,
            ),
            "memory_type": "correction",
            "content_masked": original_masked,
            "structured_payload": {
                "action": str(follow_up_action),
                "semantic_state": dict(semantic_state or {}),
                "untrusted_text": True,
            },
            "sql_fingerprint": None,
            "verification_status": "confirmed",
        })
    return candidates


def lexical_relevance(query: str, text: str) -> float:
    query_tokens = _lexical_tokens(query)
    text_tokens = _lexical_tokens(text)
    if not query_tokens or not text_tokens:
        return 0.0
    overlap = len(query_tokens & text_tokens)
    exact_bonus = 1.0 if str(query or "").strip().lower() in str(text or "").lower() else 0.0
    return round(overlap / len(query_tokens) + exact_bonus, 6)


def rrf_fuse(
    rankings: Mapping[str, Sequence[str]],
    *,
    rrf_k: int,
    limit: int,
) -> tuple[list[str], list[dict]]:
    scores: dict[str, float] = {}
    provenance: dict[str, dict] = {}
    for source, identifiers in rankings.items():
        for rank, memory_id in enumerate(identifiers, start=1):
            scores[memory_id] = scores.get(memory_id, 0.0) + 1.0 / (
                max(1, int(rrf_k)) + rank
            )
            provenance.setdefault(memory_id, {
                "memory_id": memory_id,
                "sources": [],
                "ranks": {},
            })
            provenance[memory_id]["sources"].append(source)
            provenance[memory_id]["ranks"][source] = rank
    ordered = sorted(scores, key=lambda value: (-scores[value], value))[:max(1, int(limit))]
    rows = []
    for rank, memory_id in enumerate(ordered, start=1):
        rows.append({
            **provenance[memory_id],
            "fusion_rank": rank,
            "fusion_score": round(scores[memory_id], 8),
        })
    return ordered, rows


def build_working_payload(
    *,
    semantic_state: Mapping | None,
    rolling_summary: str,
    recent_message_ids: Sequence[int],
    last_success_question: str,
    settings: ConversationMemorySettings = DEFAULT_MEMORY_SETTINGS,
) -> dict:
    payload = {
        "structured_state": dict(semantic_state or {}),
        "rolling_summary": _mask_free_text(
            rolling_summary,
            policy=settings.masking_policy,
            max_chars=settings.memory_content_max_chars,
        ),
        "recent_message_ids": [int(value) for value in recent_message_ids][
            -settings.recent_message_limit:
        ],
        "last_success_question": _mask_free_text(
            last_success_question,
            policy=settings.masking_policy,
            max_chars=settings.memory_content_max_chars,
        ),
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    payload["source_hash"] = "sha256:" + hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    return payload


def new_empty_memory_context(
    *,
    evidence: dict | None = None,
    settings: ConversationMemorySettings = DEFAULT_MEMORY_SETTINGS,
) -> MemoryContext:
    return {
        "schema_version": settings.context_schema_version,
        "working_state": {},
        "recent_messages": [],
        "selected_summaries": [],
        "confirmed_preferences": [],
        "confirmed_profile_facts": [],
        "long_term_goals": [],
        "relevant_episodes": [],
        "provenance": [],
        "omitted_counts": {},
        "evidence": dict(evidence or {}),
    }


def iso_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )
