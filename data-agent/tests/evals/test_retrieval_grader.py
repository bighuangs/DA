import pytest

from evals.contracts import RetrievedEvidence, TargetResponse
from evals.datasets import (
    Answerability,
    ClaimRecord,
    DatasetCase,
    RelevanceJudgment,
    ReviewStatus,
)
from evals.graders import GradeContext, RetrievalGrader


def test_retrieval_grader_reports_deterministic_qrel_metrics() -> None:
    case = DatasetCase(
        case_id="case-1",
        dataset_id="rag-set",
        dataset_version="v1",
        split="test",
        query="问题",
        answerability=Answerability.ANSWERABLE,
    )
    qrels = (
        RelevanceJudgment(
            case_id="case-1",
            document_id="doc-1",
            chunk_id="chunk-1",
            relevance_grade=3,
            review_status=ReviewStatus.ADJUDICATED,
        ),
        RelevanceJudgment(
            case_id="case-1",
            document_id="doc-2",
            chunk_id="chunk-2",
            relevance_grade=1,
            review_status=ReviewStatus.ADJUDICATED,
        ),
    )
    response = TargetResponse(
        answer="答案",
        retrieved=(
            RetrievedEvidence("doc-x", "chunk-x", rank=1),
            RetrievedEvidence("doc-1", "chunk-1", rank=2),
        ),
    )
    context = GradeContext(
        evaluation_id="eval-1",
        case=case,
        response=response,
        qrels=qrels,
        claims=(
            ClaimRecord(
                case_id="case-1",
                claim_id="claim-1",
                claim_text="答案",
                required=True,
                support_document_id="doc-1",
                support_chunk_id="chunk-1",
            ),
        ),
    )

    results = {item.metric_name: item.value for item in RetrievalGrader(k=2).grade(context)}

    assert results["precision_at_2"] == pytest.approx(0.5)
    assert results["recall_at_2"] == pytest.approx(0.5)
    assert results["hit_at_2"] == pytest.approx(1.0)
    assert results["mrr_at_2"] == pytest.approx(0.5)
