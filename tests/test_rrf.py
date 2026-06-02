import sys
import os
import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from retrievals.rrf import reciprocal_rank_fusion, reciprocal_rank_fusion_multi


def test_normal_fusion():
    """
    Test standard fusion between vector and keyword results as in the example.
    """
    vector_results = [
        {"chunk_id": "A"},
        {"chunk_id": "B"},
        {"chunk_id": "C"}
    ]
    keyword_results = [
        {"chunk_id": "C"},
        {"chunk_id": "A"},
        {"chunk_id": "D"}
    ]

    results = reciprocal_rank_fusion(vector_results, keyword_results, k=60)

    
    assert len(results) == 4
    assert results[0]["chunk_id"] == "A"
    assert results[1]["chunk_id"] == "C"
    assert results[2]["chunk_id"] == "B"
    assert results[3]["chunk_id"] == "D"

    
    
    assert results[0]["rrf_score"] == pytest.approx(0.032522, abs=1e-5)
    
    assert results[1]["rrf_score"] == pytest.approx(0.032264, abs=1e-5)
    
    assert results[2]["rrf_score"] == pytest.approx(0.016129, abs=1e-5)
    
    assert results[3]["rrf_score"] == pytest.approx(0.015873, abs=1e-5)


def test_empty_vector_results():
    """
    Test fusion when vector results list is empty or None.
    """
    keyword_results = [
        {"chunk_id": "C"},
        {"chunk_id": "A"}
    ]
    
    
    res1 = reciprocal_rank_fusion([], keyword_results, k=60)
    assert len(res1) == 2
    assert res1[0]["chunk_id"] == "C"
    assert res1[0]["rrf_score"] == pytest.approx(1 / 61)
    assert res1[1]["chunk_id"] == "A"
    assert res1[1]["rrf_score"] == pytest.approx(1 / 62)

    
    res2 = reciprocal_rank_fusion(None, keyword_results, k=60)
    assert len(res2) == 2
    assert res2[0]["chunk_id"] == "C"
    assert res2[1]["chunk_id"] == "A"


def test_empty_keyword_results():
    """
    Test fusion when keyword results list is empty or None.
    """
    vector_results = [
        {"chunk_id": "B"},
        {"chunk_id": "A"}
    ]

    
    res1 = reciprocal_rank_fusion(vector_results, [], k=60)
    assert len(res1) == 2
    assert res1[0]["chunk_id"] == "B"
    assert res1[0]["rrf_score"] == pytest.approx(1 / 61)
    assert res1[1]["chunk_id"] == "A"
    assert res1[1]["rrf_score"] == pytest.approx(1 / 62)

    
    res2 = reciprocal_rank_fusion(vector_results, None, k=60)
    assert len(res2) == 2
    assert res2[0]["chunk_id"] == "B"
    assert res2[1]["chunk_id"] == "A"


def test_both_empty():
    """
    Test fusion when both result lists are empty or None.
    """
    assert reciprocal_rank_fusion([], [], k=60) == []
    assert reciprocal_rank_fusion(None, None, k=60) == []


def test_duplicate_documents_in_same_list():
    """
    Test that duplicate entries in the same result list are ignored,
    keeping only the first (highest rank) occurrence.
    """
    vector_results = [
        {"chunk_id": "A"},
        {"chunk_id": "A"},  
        {"chunk_id": "B"}
    ]
    keyword_results = [
        {"chunk_id": "B"}
    ]

    results = reciprocal_rank_fusion(vector_results, keyword_results, k=60)

    
    
    
    assert len(results) == 2
    assert results[0]["chunk_id"] == "B"
    assert results[0]["rrf_score"] == pytest.approx(1 / 61 + 1 / 62)
    assert results[1]["chunk_id"] == "A"
    assert results[1]["rrf_score"] == pytest.approx(1 / 61)


def test_unique_documents():
    """
    Test that unique documents appearing in only one ranking source
    are still preserved in the merged results.
    """
    vector_results = [
        {"chunk_id": "OnlyInVector"}
    ]
    keyword_results = [
        {"chunk_id": "OnlyInKeyword"}
    ]

    results = reciprocal_rank_fusion(vector_results, keyword_results, k=60)
    assert len(results) == 2
    assert {r["chunk_id"] for r in results} == {"OnlyInVector", "OnlyInKeyword"}
    assert results[0]["rrf_score"] == pytest.approx(1 / 61)
    assert results[1]["rrf_score"] == pytest.approx(1 / 61)


def test_multi_list_fusion():
    """
    Test that reciprocal_rank_fusion_multi successfully combines more than 2 result lists.
    """
    list_1 = [{"chunk_id": "A"}, {"chunk_id": "B"}]
    list_2 = [{"chunk_id": "B"}, {"chunk_id": "C"}]
    list_3 = [{"chunk_id": "C"}, {"chunk_id": "A"}]

    results = reciprocal_rank_fusion_multi([list_1, list_2, list_3], k=60)

    
    
    
    assert len(results) == 3
    
    for r in results:
        assert r["rrf_score"] == pytest.approx(1 / 61 + 1 / 62)


def test_malformed_entries():
    """
    Test that malformed entries (non-dicts, missing chunk_ids, empty string chunk_ids)
    are gracefully ignored.
    """
    vector_results = [
        "not a dict",
        {"chunk_id": "A"},
        {"no_chunk_id": "val"},
        {"chunk_id": 123},  
        {"chunk_id": "  "},  
        {"chunk_id": "B"}
    ]
    keyword_results = [
        {"chunk_id": "B"}
    ]

    results = reciprocal_rank_fusion(vector_results, keyword_results, k=60)  # type: ignore[arg-type]
    
    
    
    
    assert len(results) == 2
    assert results[0]["chunk_id"] == "B"
    assert results[0]["rrf_score"] == pytest.approx(1 / 61 + 1 / 62)
    assert results[1]["chunk_id"] == "A"
    assert results[1]["rrf_score"] == pytest.approx(1 / 61)


def test_invalid_k():
    """
    Test that ValueError is raised when k is <= 0.
    """
    with pytest.raises(ValueError, match="Constant k must be greater than 0"):
        reciprocal_rank_fusion([], [], k=0)

    with pytest.raises(ValueError, match="Constant k must be greater than 0"):
        reciprocal_rank_fusion([], [], k=-5)
