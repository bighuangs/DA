from pathlib import Path

import pytest

from evals.datasets import DatasetLoader, DatasetValidationError
from evals.datasets.migrate_legacy_rag import render_legacy_rag_dataset
from evals.paths import RAG_V1_DATASET_ROOT


CASES_HEADER = (
    "case_id,dataset_id,dataset_version,split,query,answerability,"
    "gold_answer,language,custom_axis\n"
)
QRELS_HEADER = (
    "case_id,document_id,chunk_id,relevance_grade,review_status,"
    "evidence_start,evidence_end\n"
)
CLAIMS_HEADER = (
    "case_id,claim_id,claim_text,required,support_document_id,"
    "support_chunk_id,importance\n"
)


def _write_dataset(
    root: Path,
    *,
    cases: str,
    qrels: str,
    claims: str,
) -> None:
    (root / "cases.csv").write_text(CASES_HEADER + cases, encoding="utf-8")
    (root / "relevance-judgments.csv").write_text(
        QRELS_HEADER + qrels, encoding="utf-8"
    )
    (root / "claims.csv").write_text(CLAIMS_HEADER + claims, encoding="utf-8")


def test_loads_normalized_tables_and_preserves_extension_columns(tmp_path: Path) -> None:
    _write_dataset(
        tmp_path,
        cases="case-1,rag-set,v1,test,什么是净利润?,answerable,收入减成本,zh,finance\n",
        qrels="case-1,doc-1,chunk-1,3,adjudicated,10,30\n",
        claims="case-1,claim-1,净利润等于收入减成本,true,doc-1,chunk-1,1.0\n",
    )

    bundle = DatasetLoader().load(tmp_path)

    assert bundle.dataset_id == "rag-set"
    assert bundle.dataset_version == "v1"
    assert bundle.cases[0].attributes["custom_axis"] == "finance"
    assert bundle.qrels_for("case-1")[0].relevance_grade == 3
    assert bundle.claims_for("case-1")[0].support_key == ("doc-1", "chunk-1")


def test_rejects_orphan_qrel(tmp_path: Path) -> None:
    _write_dataset(
        tmp_path,
        cases="case-1,rag-set,v1,test,问题?,answerable,答案,zh,base\n",
        qrels="case-missing,doc-1,chunk-1,3,reviewed,,\n",
        claims="case-1,claim-1,答案,true,doc-1,chunk-1,1\n",
    )

    with pytest.raises(DatasetValidationError, match="不存在的 case_id"):
        DatasetLoader().load(tmp_path)


def test_rejects_duplicate_qrel_key(tmp_path: Path) -> None:
    _write_dataset(
        tmp_path,
        cases="case-1,rag-set,v1,test,问题?,answerable,答案,zh,base\n",
        qrels=(
            "case-1,doc-1,chunk-1,3,reviewed,,\n"
            "case-1,doc-1,chunk-1,2,adjudicated,,\n"
        ),
        claims="case-1,claim-1,答案,true,doc-1,chunk-1,1\n",
    )

    with pytest.raises(DatasetValidationError, match="重复 Evidence Key"):
        DatasetLoader().load(tmp_path)


def test_rejects_claim_support_not_present_in_qrels(tmp_path: Path) -> None:
    _write_dataset(
        tmp_path,
        cases="case-1,rag-set,v1,test,问题?,answerable,答案,zh,base\n",
        qrels="case-1,doc-1,chunk-1,3,reviewed,,\n",
        claims="case-1,claim-1,答案,true,doc-2,chunk-2,1\n",
    )

    with pytest.raises(DatasetValidationError, match="支持 Evidence 不存在"):
        DatasetLoader().load(tmp_path)


def test_rejects_positive_qrel_for_unanswerable_case(tmp_path: Path) -> None:
    _write_dataset(
        tmp_path,
        cases="case-1,rag-set,v1,test,未知问题?,unanswerable,,zh,safety\n",
        qrels="case-1,doc-1,chunk-1,1,reviewed,,\n",
        claims="",
    )

    with pytest.raises(DatasetValidationError, match="不可回答 Case"):
        DatasetLoader().load(tmp_path)


def test_allows_retrieval_only_case_without_gold_claim(tmp_path: Path) -> None:
    (tmp_path / "cases.csv").write_text(
        CASES_HEADER.replace("custom_axis", "evaluation_scope")
        + "case-1,rag-set,v1,test,检索问题?,answerable,,zh,retrieval_only\n",
        encoding="utf-8",
    )
    (tmp_path / "relevance-judgments.csv").write_text(
        QRELS_HEADER + "case-1,doc-1,chunk-1,3,draft,,\n",
        encoding="utf-8",
    )
    (tmp_path / "claims.csv").write_text(
        CLAIMS_HEADER,
        encoding="utf-8",
    )

    bundle = DatasetLoader().load(tmp_path)

    assert bundle.claims == ()


def test_rejects_answerable_case_without_claim_or_retrieval_only_scope(
    tmp_path: Path,
) -> None:
    _write_dataset(
        tmp_path,
        cases="case-1,rag-set,v1,test,检索问题?,answerable,,zh,unscoped\n",
        qrels="case-1,doc-1,chunk-1,3,draft,,\n",
        claims="",
    )

    with pytest.raises(DatasetValidationError, match="evaluation_scope=retrieval_only"):
        DatasetLoader().load(tmp_path)


def test_committed_rag_v1_is_reproducible_and_loadable() -> None:
    rendered = render_legacy_rag_dataset()

    for filename, expected in rendered.items():
        assert (RAG_V1_DATASET_ROOT / filename).read_text(encoding="utf-8") == expected

    bundle = DatasetLoader().load(RAG_V1_DATASET_ROOT)
    assert len(bundle.cases) == 16
    assert bundle.claims == ()
    assert all(case.attributes["evaluation_scope"] == "retrieval_only" for case in bundle.cases)
    assert all(qrel.review_status.value == "draft" for qrel in bundle.qrels)
