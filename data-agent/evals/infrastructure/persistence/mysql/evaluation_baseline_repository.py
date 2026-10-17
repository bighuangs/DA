from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from evals.legacy.governance import baseline_checklist, is_explicit_nonrelease
from app.infrastructure.persistence.mysql.records.evaluation_baseline_mysql import EvaluationBaselineMySQL
from app.infrastructure.persistence.mysql.records.workspace_mysql import WorkspaceMySQL


def _now() -> datetime:
    return datetime.now().astimezone().replace(tzinfo=None)


def _run_identity(run: dict) -> tuple[str, str, str, str | None]:
    workspace_id = str(run.get("workspace_id") or "").strip()
    run_id = str(run.get("run_id") or "").strip()
    dataset_suite = str(run.get("dataset_suite") or "").strip()
    run_fingerprint = str(
        (run.get("fingerprint") or {}).get("run_fingerprint") or ""
    ).strip() or None
    missing = [
        name
        for name, value in (
            ("workspace_id", workspace_id),
            ("run_id", run_id),
            ("dataset_suite", dataset_suite),
            ("run_fingerprint", run_fingerprint),
        )
        if not value
    ]
    if missing:
        raise ValueError("baseline run 缺少身份字段：" + "、".join(missing))
    return workspace_id, run_id, dataset_suite, run_fingerprint


def _require_scope(workspace_id: str, dataset_suite: str) -> tuple[str, str]:
    normalized_workspace = str(workspace_id or "").strip()
    normalized_suite = str(dataset_suite or "").strip()
    if not normalized_workspace or not normalized_suite:
        raise ValueError("baseline 查询必须显式提供 workspace_id 与 dataset_suite")
    return normalized_workspace, normalized_suite


def _assert_registration_matches(
    existing: dict,
    *,
    dataset_suite: str,
    artifact: str | None,
    run_fingerprint: str | None,
    compare_artifact: bool = True,
) -> None:
    conflicts: list[str] = []
    if str(existing.get("dataset_suite") or "") != dataset_suite:
        conflicts.append("dataset_suite")
    if str(existing.get("run_fingerprint") or "") != str(run_fingerprint or ""):
        conflicts.append("run_fingerprint")
    if compare_artifact and str(existing.get("artifact") or "") != str(artifact or ""):
        conflicts.append("artifact")
    if conflicts:
        raise ValueError(
            f"run_id {existing.get('run_id')} 已登记但身份冲突："
            + "、".join(conflicts)
        )


class EvaluationBaselineRepository:
    """Database-backed baseline governance with workspace row locking."""

    def __init__(self, session: AsyncSession):
        self.session = session

    @staticmethod
    def view(item: EvaluationBaselineMySQL | None) -> dict | None:
        if item is None:
            return None
        return {
            "run_id": item.run_id, "workspace_id": item.workspace_id,
            "dataset_suite": item.dataset_suite,
            "artifact": item.artifact, "status": item.status, "eligible": bool(item.eligible),
            "checklist": dict(item.checklist or {}), "run_fingerprint": item.run_fingerprint,
            "reviewed_by": item.reviewed_by, "review_note": item.review_note,
            "reviewed_at": item.reviewed_at.isoformat() if item.reviewed_at else None,
            "superseded_at": item.superseded_at.isoformat() if item.superseded_at else None,
            "created_at": item.created_at.isoformat() if item.created_at else None,
        }

    async def get(self, workspace_id: str, run_id: str) -> dict | None:
        item = (await self.session.execute(
            select(EvaluationBaselineMySQL)
            .where(EvaluationBaselineMySQL.workspace_id == workspace_id)
            .where(EvaluationBaselineMySQL.run_id == run_id)
        )).scalars().first()
        return self.view(item)

    async def approved(self, workspace_id: str, dataset_suite: str) -> dict | None:
        workspace_id, dataset_suite = _require_scope(workspace_id, dataset_suite)
        item = (await self.session.execute(
            select(EvaluationBaselineMySQL)
            .where(EvaluationBaselineMySQL.workspace_id == workspace_id)
            .where(EvaluationBaselineMySQL.dataset_suite == dataset_suite)
            .where(EvaluationBaselineMySQL.status == "approved")
            .order_by(
                EvaluationBaselineMySQL.reviewed_at.desc(),
                EvaluationBaselineMySQL.id.desc(),
            )
        )).scalars().first()
        return self.view(item)

    async def approved_all(self, workspace_id: str) -> list[dict]:
        workspace_id = str(workspace_id or "").strip()
        if not workspace_id:
            raise ValueError("baseline 查询必须显式提供 workspace_id")
        items = list((await self.session.execute(
            select(EvaluationBaselineMySQL)
            .where(EvaluationBaselineMySQL.workspace_id == workspace_id)
            .where(EvaluationBaselineMySQL.status == "approved")
            .order_by(
                EvaluationBaselineMySQL.dataset_suite.asc(),
                EvaluationBaselineMySQL.reviewed_at.desc(),
                EvaluationBaselineMySQL.id.desc(),
            )
        )).scalars().all())
        return [self.view(item) for item in items]

    async def register_candidate(self, run: dict, artifact: str, *, commit: bool = True) -> dict:
        if is_explicit_nonrelease(run):
            raise ValueError("non-release 评测制品不得注册 baseline candidate")
        if run.get("baseline_eligible") is not True:
            raise ValueError("baseline_eligible 必须显式为 true 才可注册 candidate")
        workspace_id, run_id, dataset_suite, run_fingerprint = _run_identity(run)
        artifact = str(artifact or "").strip()
        if not artifact:
            raise ValueError("baseline candidate 必须绑定 artifact")
        existing = await self.get(workspace_id, run_id)
        if existing:
            _assert_registration_matches(
                existing,
                dataset_suite=dataset_suite,
                artifact=artifact,
                run_fingerprint=run_fingerprint,
            )
            return existing
        full_count = int(((run.get("fingerprint") or {}).get("sample_contract") or {}).get("total") or 0)
        checklist = baseline_checklist(run, full_count)
        if not checklist["eligible"]:
            failed = [
                key for key, value in checklist["checks"].items() if not value
            ]
            raise ValueError(
                "该 run 未满足 baseline candidate 准入：" + "、".join(failed)
            )
        item = EvaluationBaselineMySQL(
            workspace_id=workspace_id, dataset_suite=dataset_suite,
            run_id=run_id, artifact=artifact,
            status="candidate", eligible=checklist["eligible"], checklist=checklist["checks"],
            run_fingerprint=run_fingerprint,
        )
        try:
            # A savepoint keeps the outer session usable when another worker wins
            # the workspace/run unique-key race between the initial read and insert.
            async with self.session.begin_nested():
                self.session.add(item)
                await self.session.flush()
        except IntegrityError:
            existing = await self.get(workspace_id, run_id)
            if existing is None:
                raise
            _assert_registration_matches(
                existing,
                dataset_suite=dataset_suite,
                artifact=artifact,
                run_fingerprint=run_fingerprint,
            )
            if commit:
                await self.session.commit()
            return existing
        if commit:
            await self.session.commit()
            # MySQL server defaults (notably created_at) may remain unloaded or
            # become expired after commit even when the session factory uses
            # expire_on_commit=False.  Refresh explicitly inside the async
            # greenlet before the synchronous serializer reads ORM attributes.
            await self.session.refresh(item)
        return self.view(item)

    async def review(self, run: dict, *, decision: str, reviewer: str,
                     note: str = "", commit: bool = True) -> dict:
        if is_explicit_nonrelease(run):
            raise ValueError("non-release 评测制品不得进入 baseline review")
        if run.get("baseline_eligible") is not True:
            raise ValueError("baseline_eligible 必须显式为 true 才可进入 review")
        if decision not in {"approve", "reject"}:
            raise ValueError("decision 只允许 approve 或 reject")
        full_count = int(((run.get("fingerprint") or {}).get("sample_contract") or {}).get("total") or 0)
        checklist = baseline_checklist(run, full_count)
        if decision == "approve" and not checklist["eligible"]:
            failed = [key for key, value in checklist["checks"].items() if not value]
            raise ValueError("该 run 未满足 baseline 准入：" + "、".join(failed))

        workspace_id, run_id, dataset_suite, run_fingerprint = _run_identity(run)
        # The workspace row is stable even before the first candidate exists, so it
        # serializes approvals for the same tenant across workers and processes.
        locked_workspace = (await self.session.execute(
            select(WorkspaceMySQL.id)
            .where(WorkspaceMySQL.id == workspace_id)
            .with_for_update()
        )).scalar_one_or_none()
        if locked_workspace is None:
            raise ValueError(f"baseline workspace 不存在：{workspace_id}")

        collision = (await self.session.execute(
            select(EvaluationBaselineMySQL)
            .where(EvaluationBaselineMySQL.workspace_id == workspace_id)
            .where(EvaluationBaselineMySQL.run_id == run_id)
            .with_for_update()
        )).scalars().first()
        if collision is not None:
            _assert_registration_matches(
                self.view(collision),
                dataset_suite=dataset_suite,
                artifact=run.get("_artifact"),
                run_fingerprint=run_fingerprint,
                compare_artifact=run.get("_artifact") is not None,
            )

        # Lock only this suite's candidates/baselines before superseding its
        # previous approval. Other difficulty suites remain independently approved.
        rows = list((await self.session.execute(
            select(EvaluationBaselineMySQL)
            .where(EvaluationBaselineMySQL.workspace_id == workspace_id)
            .where(EvaluationBaselineMySQL.dataset_suite == dataset_suite)
            .with_for_update()
        )).scalars().all())
        item = next((row for row in rows if row.run_id == run_id), None)
        if item is None:
            item = EvaluationBaselineMySQL(
                workspace_id=workspace_id, dataset_suite=dataset_suite,
                run_id=run_id, artifact=run.get("_artifact"),
                status="candidate", eligible=checklist["eligible"], checklist=checklist["checks"],
                run_fingerprint=run_fingerprint,
            )
            self.session.add(item)
            await self.session.flush()
            rows.append(item)
        now = _now()
        if decision == "approve":
            for row in rows:
                if row.id != item.id and row.status == "approved":
                    row.status = "superseded"
                    row.superseded_at = now
        item.status = "approved" if decision == "approve" else "rejected"
        item.eligible = checklist["eligible"]
        item.checklist = checklist["checks"]
        item.reviewed_at = now
        item.reviewed_by = reviewer[:128]
        item.review_note = note[:500]
        item.run_fingerprint = run_fingerprint
        if commit:
            await self.session.commit()
            await self.session.refresh(item)
        return self.view(item)
