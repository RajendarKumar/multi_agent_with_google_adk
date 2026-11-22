"""
Unit tests for the Ingestion Indexer and BM25 sparse vector calculation.
"""

from ingestion.indexer import HybridIndexer


def test_bm25_sparse_vector_generation():
    indexer = HybridIndexer()
    sample_text = "Google ADK agentic RAG platform for document ingestion and search"

    sparse_vec = indexer._bm25_sparse(sample_text)

    assert "indices" in sparse_vec
    assert "values" in sparse_vec
    assert len(sparse_vec["indices"]) == len(sparse_vec["values"])
    assert len(sparse_vec["indices"]) > 0

    # Term weights should be positive float values
    for val in sparse_vec["values"]:
        assert val > 0.0


def test_bm25_sparse_reproducibility():
    indexer = HybridIndexer()
    text = "Machine learning vector search"

    vec1 = indexer._bm25_sparse(text)
    vec2 = indexer._bm25_sparse(text)

    assert vec1["indices"] == vec2["indices"]
    assert vec1["values"] == vec2["values"]
