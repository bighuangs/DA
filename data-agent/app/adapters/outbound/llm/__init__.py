"""LLM 出站适配器。"""

from app.adapters.outbound.llm.analysis_model import LLMAnalysisModel
from app.adapters.outbound.llm.routing import Nl2SqlRouteCatalog

__all__ = ["LLMAnalysisModel", "Nl2SqlRouteCatalog"]
