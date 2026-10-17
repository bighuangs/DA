"""Read-only access to the latest TODO-40 RAG evaluation artifact."""

from __future__ import annotations

import json

from evals.legacy.rag_recall import LATEST_PATH, evaluate_live_run, validate_contract
from evals.paths import RAG_RUNS_ROOT, REPORTS_ROOT


HYBRID_REPORT_PATH = REPORTS_ROOT / "hybrid_retrieval_latest.json"
POINT_PROJECTION_REPORT_PATH = REPORTS_ROOT / "point_projection_latest.json"
EMBEDDING_CAPACITY_REPORT_PATH = REPORTS_ROOT / "embedding_capacity_latest.json"
QUERY_SIGNAL_REPORT_PATH = REPORTS_ROOT / "query_signal_ablation_latest.json"


class RAGEvaluationService:
    def latest(self, workspace_id: str | None = None) -> dict:
        report = None
        if LATEST_PATH.exists():
            try:
                report = json.loads(LATEST_PATH.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        contract = validate_contract()
        if report is None:
            report = {
                "available": False, "generated_at": None, "source": "contract_only",
                "live_index_evaluated": False, "contract": contract,
                "summary": {"cases": contract["total"], "gate_passed": False},
                "baseline": None, "tuned": None, "comparison": None,
            }
        live = None
        # RAG 页面只接受专用 live runner 的同协议制品。普通 NL2SQL run
        # 不具备完整 RAG 分母，禁止再作为兼容事实源。
        if RAG_RUNS_ROOT.exists():
            for path in sorted(
                RAG_RUNS_ROOT.glob("rag_run_*.json"),
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            ):
                try:
                    run = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if workspace_id and str(run.get("workspace_id") or "") != str(workspace_id):
                    continue
                if not bool(run.get("valid")) or int(run.get("n") or 0) < int(contract["total"]):
                    continue
                live = evaluate_live_run(run)
                if live.get("available"):
                    break
                live = None
        report["live"] = live or {
            "available": False,
            "message": "当前工作空间尚无覆盖 RAG 契约的有效 full run。",
        }
        hybrid = None
        if HYBRID_REPORT_PATH.exists():
            try:
                candidate = json.loads(HYBRID_REPORT_PATH.read_text(encoding="utf-8"))
                if (
                    not workspace_id
                    or str(candidate.get("workspace_id") or "") == str(workspace_id)
                ):
                    hybrid = candidate
            except (OSError, json.JSONDecodeError):
                pass
        report["hybrid"] = hybrid or {
            "available": False,
            "message": "当前工作空间尚未生成 TODO-56 BGE-M3 混合检索报告。",
        }
        point_projection = None
        if POINT_PROJECTION_REPORT_PATH.exists():
            try:
                candidate = json.loads(
                    POINT_PROJECTION_REPORT_PATH.read_text(encoding="utf-8")
                )
                if (
                    not workspace_id
                    or str(candidate.get("workspace_id") or "") == str(workspace_id)
                ):
                    point_projection = candidate
            except (OSError, json.JSONDecodeError):
                pass
        report["point_projection"] = point_projection or {
            "available": False,
            "message": "当前工作空间尚未生成 TODO-57 Point 投影消融报告。",
        }
        embedding_capacity = None
        if EMBEDDING_CAPACITY_REPORT_PATH.exists():
            try:
                embedding_capacity = json.loads(
                    EMBEDDING_CAPACITY_REPORT_PATH.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                pass
        report["embedding_capacity"] = embedding_capacity or {
            "available": False,
            "message": "尚未生成 TODO-60 Embedding 调度容量报告。",
        }
        query_signals = None
        if QUERY_SIGNAL_REPORT_PATH.exists():
            try:
                query_signals = json.loads(
                    QUERY_SIGNAL_REPORT_PATH.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                pass
        report["query_signals"] = query_signals or {
            "available": False,
            "message": "尚未生成 TODO-61 查询信号消融报告。",
        }
        return report
