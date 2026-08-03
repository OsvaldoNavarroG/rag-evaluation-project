from rag.attribution import (
    extract_citations,
    extract_cited_claims,
    evaluate_citation_precision,
    evaluate_claim_attribution,
)


def test_duplicate_citations_are_counted_once() -> None:
    answer = "Claim A [0]. Claim B [0][1]."

    citations = extract_citations(answer=answer)
    unique_citations = list(dict.fromkeys(citations))

    assert citations == [0, 0, 1]
    assert unique_citations == [0, 1]


def test_citation_precision_uses_unique_indices(monkeypatch) -> None:
    answer = "Claim A [0]. Claim B [0][1]."
    chunks = ["supporting chunk", "unsupported chunk"]
    support = {"supporting chunk": True, "unsupported chunk": False}

    monkeypatch.setattr(
        "rag.attribution.chunk_supports_answer", lambda answer, chunk: support[chunk]
    )

    result = evaluate_citation_precision(answer=answer, chunks=chunks)

    assert result["citation_precision"] == 0.5


def test_extract_cited_claims_multiple_sentences() -> None:
    answer = (
        "Computer vision is the broader field [1]. "
        "Image recognition classifies images [2]."
    )

    result = extract_cited_claims(answer=answer)

    assert result == [
        {"claim": "Computer vision is the broader field.", "citation_indices": [1]},
        {"claim": "Image recognition classifies images.", "citation_indices": [2]},
    ]


def test_extract_cited_claims_multiple_citations() -> None:
    answer = "Image recognition is a computer-vision task [2][4]."

    result = extract_cited_claims(answer=answer)

    assert result == [
        {
            "claim": "Image recognition is a computer-vision task.",
            "citation_indices": [2, 4],
        }
    ]


def test_extract_cited_claims_deduplicates_citations() -> None:
    answer = "The claim is supported by several passages [3][1][3][2]."
    result = extract_cited_claims(answer=answer)
    assert result[0]["citation_indices"] == [3, 1, 2]


def test_extract_cited_claims_retains_uncited_claims() -> None:
    answer = (
        "Computer vision interprets visual information [1]. " "It has many applications"
    )
    result = extract_cited_claims(answer=answer)
    assert result == [
        {
            "claim": "Computer vision interprets visual information.",
            "citation_indices": [1],
        },
        {"claim": "It has many applications", "citation_indices": []},
    ]


def test_extract_cited_claims_empty_answer() -> None:
    assert extract_cited_claims("") == []
    assert extract_cited_claims("     ") == []


def _stub_support(claim: str, chunk: str) -> bool:
    # Deterministic support rule for tests: a chunk supports a claim iff it
    # contains the token "yes". Keeps tests isolated from the fuzzy lexical
    # overlap heuristic so they test claim-attribution logic.
    return "yes" in chunk


def test_claim_attribution_all_cited_and_supported() -> None:
    answer = "Claim A [0]. Claim B [1]."
    chunks = ["yes chunk", "yes chunk"]
    result = evaluate_claim_attribution(
        answer=answer, chunks=chunks, support_fn=_stub_support
    )
    assert result["total_claims"] == 2
    assert result["cited_claims"] == 2
    assert result["supported_claims"] == 2
    assert result["claim_precision"] == 1.0
    assert result["claim_coverage"] == 1.0


def test_claim_attribution_wrong_citation_lowers_precision() -> None:
    answer = "Claim A [0]. Claim B [1]."
    chunks = ["yes chunk", "no chunk"]  # claim B cites and unsupporting chunk
    result = evaluate_claim_attribution(
        answer=answer, chunks=chunks, support_fn=_stub_support
    )
    assert result["cited_claims"] == 2
    assert result["supported_claims"] == 1
    assert result["claim_precision"] == 0.5
    assert result["claim_coverage"] == 1.0


def test_claim_attribution_uncited_claim_lowers_coverage() -> None:
    answer = "Claim A [0]. Claim B has no citation."
    chunks = ["yes chunk"]
    result = evaluate_claim_attribution(
        answer=answer, chunks=chunks, support_fn=_stub_support
    )
    assert result["total_claims"] == 2
    assert result["cited_claims"] == 1
    assert result["claim_precision"] == 1.0  # over cited claims only
    assert result["claim_coverage"] == 0.5


def test_claim_attribution_out_of_range_index_is_unsupported() -> None:
    answer = "Claim A [5]."
    chunks = ["yes chunk"]
    result = evaluate_claim_attribution(
        answer=answer, chunks=chunks, support_fn=_stub_support
    )
    assert result["cited_claims"] == 1
    assert result["supported_claims"] == 0
    assert result["claim_precision"] == 0.0


def test_claim_attribution_no_cited_claims_precision_is_none() -> None:
    answer = "just a bare assertion with no citation."
    chunks = ["yes chunk"]
    result = evaluate_claim_attribution(
        answer=answer, chunks=chunks, support_fn=_stub_support
    )
    assert result["cited_claims"] == 0
    assert result["claim_precision"] is None
    assert result["claim_coverage"] == 0.0


def test_claim_attribution_empty_answer_is_all_none() -> None:
    result = evaluate_claim_attribution(
        answer="", chunks=["yes chunk"], support_fn=_stub_support
    )
    assert result["total_claims"] == 0
    assert result["claim_precision"] is None
    assert result["claim_coverage"] is None

def test_extract_citations_ignore_parenthesized_numbers()-> None:
    # Years, percentages, and counts in parentheses must NOT be read
    # as citations. Only [i] bracket form counts.
    answer = "CNNs emerged in the 1990s (1998) and hit 95% accuracy (98) [3]."
    assert extract_citations(answer=answer) == [3]