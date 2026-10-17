"""API 会话目录与 transcript 水合服务。"""
from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from typing import Sequence

from app.application.ports.conversation import ConversationHistoryStore
from app.application.settings.conversation import (
    DEFAULT_HISTORY_SETTINGS,
    ConversationHistorySettings,
)
from app.domain.access.policy import PolicySnapshot
from app.domain.privacy.masking import MaskingRule, mask_rows
from app.domain.query.chart_specs import build_deterministic_chart
from app.domain.query.safety.sql_security import extract_table_names


class InvalidConversationCursor(ValueError):
    pass


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _encode_cursor(value: datetime, row_id: int) -> str:
    payload = json.dumps(
        {"at": _iso(value), "id": int(row_id)},
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _decode_cursor(value: str | None) -> tuple[datetime | None, int | None]:
    text = str(value or "").strip()
    if not text:
        return None, None
    if len(text) > 512:
        raise InvalidConversationCursor("会话游标无效或已过期")
    try:
        padded = text + "=" * (-len(text) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
        at = datetime.fromisoformat(str(payload["at"]).replace("Z", "+00:00"))
        return at.astimezone(timezone.utc).replace(tzinfo=None), int(payload["id"])
    except Exception as exc:
        raise InvalidConversationCursor("会话游标无效或已过期") from exc


def _session_payload(row) -> dict:
    return {
        "chat_id": row.chat_id,
        "title": row.title,
        "preview": row.preview,
        "status": row.status,
        "last_turn_status": row.last_turn_status,
        "turn_count": int(row.turn_count or 0),
        "created_at": _iso(row.created_at),
        "last_message_at": _iso(row.last_message_at),
        "updated_at": _iso(row.updated_at),
        "archived_at": _iso(row.archived_at),
        "context_reset_at": _iso(row.context_reset_at),
        "version": int(row.version or 0),
    }


class ConversationHistoryService:
    def __init__(
        self,
        store: ConversationHistoryStore,
        *,
        access_policy: PolicySnapshot,
        masking_rules: Sequence[MaskingRule] = (),
        settings: ConversationHistorySettings = DEFAULT_HISTORY_SETTINGS,
    ):
        self.repo = store
        self.access_policy = access_policy
        self.masking_rules = tuple(masking_rules)
        self.settings = settings

    async def list_sessions(
        self,
        *,
        status: str = "active",
        limit: int | None = None,
        cursor: str | None = None,
    ) -> dict:
        if status not in {"active", "archived"}:
            raise ValueError("status 只支持 active 或 archived")
        before_at, before_id = _decode_cursor(cursor)
        page_size = max(
            1,
            min(
                int(limit or self.settings.directory_page_size),
                50,
            ),
        )
        rows = await self.repo.list_sessions(
            status=status,
            limit=page_size,
            before_at=before_at,
            before_id=before_id,
        )
        has_more = len(rows) > page_size
        page = rows[:page_size]
        next_cursor = (
            _encode_cursor(page[-1].last_message_at, page[-1].id)
            if has_more and page
            else None
        )
        return {
            "items": [_session_payload(row) for row in page],
            "next_cursor": next_cursor,
            "has_more": has_more,
            "status": status,
            "retention_days": int(self.settings.retention_days),
        }

    async def get_turns(
        self,
        chat_id: str,
        *,
        limit: int | None = None,
        before: str | None = None,
    ) -> dict | None:
        conversation = await self.repo.get_session(chat_id)
        if conversation is None:
            return None
        before_at, before_id = _decode_cursor(before)
        page_size = max(
            1,
            min(
                int(limit or self.settings.transcript_page_size),
                100,
            ),
        )
        rows = await self.repo.list_turns(
            chat_id=chat_id,
            limit=page_size,
            before_at=before_at,
            before_id=before_id,
        )
        has_more = len(rows) > page_size
        page_desc = rows[:page_size]
        next_cursor = (
            _encode_cursor(page_desc[-1].created_at, page_desc[-1].id)
            if has_more and page_desc
            else None
        )
        histories = await self.repo.history_by_ids(
            row.source_history_id
            for row in page_desc
            if row.source_history_id is not None
        )
        source_contexts = await self.repo.turn_snapshots(
            row.turn_id for row in page_desc
        )
        items = [
            self._turn_payload(
                row,
                histories.get(int(row.source_history_id)),
                source_contexts.get(str(row.turn_id)),
            )
            if row.source_history_id is not None
            else self._turn_payload(
                row, None, source_contexts.get(str(row.turn_id))
            )
            for row in reversed(page_desc)
        ]
        return {
            "conversation": _session_payload(conversation),
            "items": items,
            "next_cursor": next_cursor,
            "has_more": has_more,
            "retention_days": int(self.settings.retention_days),
        }

    def _turn_payload(self, turn, history, source_context: dict | None = None) -> dict:
        unavailable: list[str] = []
        replay_fidelity = str(turn.replay_fidelity or "message_only")
        sql: str | None = None
        result_rows: list[dict] = []
        columns: list[str] = []
        row_count: int | None = None
        result_truncated = False
        chart: dict | None = None

        if turn.status == "succeeded":
            if history is None:
                if replay_fidelity != "message_only":
                    replay_fidelity = "expired"
                unavailable.extend(["sql", "result"])
            else:
                row_count = (
                    int(history.row_count)
                    if history.row_count is not None
                    else None
                )
                sql = str(history.sql or "")
                try:
                    self.access_policy.authorize_sql(sql)
                except Exception:
                    sql = None
                    unavailable.extend(["sql", "result"])
                    replay_fidelity = "redacted"

                row_filter_applies = False
                if sql is not None:
                    try:
                        row_filter_applies = any(
                            self.access_policy.table_rule(table)["row_filters"]
                            for table in extract_table_names(sql)
                        )
                    except Exception:
                        row_filter_applies = True
                if row_filter_applies:
                    unavailable.append("result")
                    replay_fidelity = "redacted"
                elif sql is not None and history.result:
                    try:
                        parsed = json.loads(history.result)
                        if not isinstance(parsed, list):
                            raise ValueError("历史结果不是行列表")
                        max_rows = max(
                            1,
                            int(self.settings.result_preview_rows),
                        )
                        max_bytes = max(
                            1024,
                            int(self.settings.result_preview_bytes),
                        )
                        candidate = parsed[:max_rows]
                        bounded: list[dict] = []
                        used_bytes = 2
                        for raw_row in candidate:
                            if not isinstance(raw_row, dict):
                                continue
                            encoded = json.dumps(
                                raw_row,
                                ensure_ascii=False,
                                default=str,
                            ).encode("utf-8")
                            if bounded and used_bytes + len(encoded) + 1 > max_bytes:
                                break
                            if not bounded and len(encoded) + 2 > max_bytes:
                                break
                            bounded.append(raw_row)
                            used_bytes += len(encoded) + 1
                        result_rows = mask_rows(
                            bounded,
                            columns=list(bounded[0].keys()) if bounded else None,
                            rules=self.masking_rules,
                            policy=self.settings.masking_policy,
                        )
                        columns = (
                            list(result_rows[0].keys()) if result_rows else []
                        )
                        result_truncated = (
                            len(result_rows) < len(parsed)
                            or len(parsed) > max_rows
                        )
                        reconstructed = build_deterministic_chart(
                            str(
                                history.standalone_question
                                or history.query
                                or turn.user_text
                                or ""
                            ),
                            result_rows,
                        )
                        if reconstructed.get("type") != "table":
                            chart = reconstructed
                    except Exception:
                        unavailable.append("result")
                        replay_fidelity = "partial"
                elif sql is not None:
                    unavailable.append("result")
                    replay_fidelity = "partial"

        if chart is None:
            unavailable.append("chart")
        unavailable.extend(["analysis", "recommendation"])
        return {
            "turn_id": turn.turn_id,
            "chat_id": turn.chat_id,
            "trace_id": turn.trace_id,
            "status": turn.status,
            "created_at": _iso(turn.created_at),
            "completed_at": _iso(turn.completed_at),
            "user_text": turn.user_text,
            "assistant_summary": turn.assistant_summary,
            "sql": sql,
            "result": {
                "columns": columns,
                "rows": result_rows,
                "row_count": row_count,
                "truncated": result_truncated,
            }
            if "result" not in unavailable
            else None,
            "chart": chart,
            "replay_fidelity": replay_fidelity,
            "unavailable_parts": list(dict.fromkeys(unavailable)),
            "data_source_context": dict(source_context or {}),
        }

    async def update_session(
        self,
        chat_id: str,
        *,
        title: str | None = None,
        status: str | None = None,
    ) -> dict | None:
        if title is not None:
            title = str(title).strip()[: int(self.settings.title_max_chars)]
            if not title:
                raise ValueError("会话标题不能为空")
        row = await self.repo.update_session(
            chat_id,
            title=title,
            status=status,
        )
        if row is None:
            return None
        await self.repo.commit()
        await self.repo.refresh(row)
        return _session_payload(row)

    async def delete_session(self, chat_id: str) -> bool:
        deleted = await self.repo.soft_delete(chat_id)
        if deleted:
            await self.repo.commit()
        return deleted

    async def begin_delete(self, chat_id: str) -> bool:
        """跨存储清理前先提交会话级隔离栅栏。"""
        deleting = await self.repo.begin_delete(chat_id)
        if deleting:
            await self.repo.commit()
        return deleting
