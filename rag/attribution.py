import re
from collections.abc import Callable
from typing import Any, TypedDict

from nltk import sent_tokenize

from rag.helpers import normalize

STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "in",
    "on",
    "of",
    "to",
    "for",
    "and",
    "or",
    "when",
    "by",
    "with",
    "through",
    "where",
    "based",
    "its",
    "it",
    "that",
    "this",
    "as",
    "such",
}


class CitedClaim(TypedDict):
    claim: str
    citation_indices: list[int]


def remove_citations(text: str) -> str:
    """Removes numeric citations such as [0] and [12] from text."""
    text_without_citations = re.sub(r"\[\d+\]", "", text)

    # Remove spaces left before punctuation: "claim ." -> "claim."
    text_without_citations = re.sub(r"\s+([.,;:!?])", r"\1", text_without_citations)
    # Normalize repeated whitespace
    return re.sub(r"\s+", " ", text_without_citations).strip()


def extract_cited_claims(answer: str) -> list[CitedClaim]:
    """
    Split an answer into sentence level claims and extracts the citations
    attached to each claim.

    Sentences without citations are retained with an empty citation list.
    This allows citation completeness to be evaluated later.
    """
    if not answer.strip():
        return []

    claims: list[CitedClaim] = []

    for sentence in sent_tokenize(answer):
        citation_indices = list(dict.fromkeys(extract_citations(answer=sentence)))
        claim_text = remove_citations(text=sentence)
        if not claim_text:
            continue

        claims.append({"claim": claim_text, "citation_indices": citation_indices})

    return claims


def extract_citations(answer: str) -> list[int]:
    """
    Extract citation indices in [i] bracket form from an answer.

    Only bracket form is recognized. Parenthesised numbers such as years
    "(2024)" for figures "(95)" are deliberately NOT treated as citations,
    since the generation prompt emits citations as [i] and matching "(\\d+)"
    would turn any parenthesised number into a spurious citation index.
    """
    bracket_matches: list = re.findall(r"\[(\d+)\]", answer)

    return [int(c) for c in bracket_matches]


def strip_citations(text: str) -> str:
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\(\d+\)", "", text)
    return text


def chunk_supports_answer(answer: str, chunk: str) -> bool:
    clean_answer: str = strip_citations(text=answer)

    answer_words: set[str] = {
        w for w in normalize(text=clean_answer).split() if w not in STOPWORDS
    }
    chunk_words: set[str] = {
        w for w in normalize(text=chunk).split() if w not in STOPWORDS
    }

    overlap = len(answer_words & chunk_words)
    coverage = overlap / max(len(answer_words), 1)

    return coverage >= 0.5


def evaluate_faithfulness(answer: str, chunks: list[str]) -> dict[str, bool]:
    """
    Evaluates whether cited chunks support the generated answer.

    Current definition:

    - The answer must contain citations.
    - All cited chunk indices must be valid.
    - Every cited chunk must individually provide sufficient lexical
      support for the answer according to 'chunk_supports_answer()'.

    Notes:
    - This is a heuristic metric based on token overlap.
    - This is stricter than standard groundedness because support is
      checked only against cited chunks.
    - Multi-claim answers may be penalized when different claims are
      supported by different citations.
    - This metric does NOT verify citation correctness at the claim level

    Returns:
    {
    "has_citations": bool,
    "valid_citations": bool,
    "faithful": bool
    }
    """
    citations: list = extract_citations(answer=answer)
    if not citations:
        return {"has_citations": False, "valid_citations": False, "faithful": False}

    for idx in citations:
        if idx < 0 or idx >= len(chunks):
            return {"has_citations": True, "valid_citations": False, "faithful": False}

        chunk: str = chunks[idx]

        if not chunk_supports_answer(answer=answer, chunk=chunk):
            return {"has_citations": True, "valid_citations": True, "faithful": False}

    return {"has_citations": True, "valid_citations": True, "faithful": True}


def evaluate_citation_precision(answer: str, chunks: list[str]) -> dict[str, Any]:
    """
    Evaluates whether the answer cites supporting chunks.

    Current definition:
    - The answer must contain citations,
    - All cited indices must be valid.
    - At least one cited chunk must support the answer.

    This is less strict than `evaluate_faithfulness()`, which requires every
    cited chunk to support the answer.

    Returns:
    {
    "has_citations": bool,
    "valid_citations": bool,
    "citation_precision": float
    }
    """
    citations: list[int] = list(dict.fromkeys(extract_citations(answer=answer)))

    if not citations:
        return {
            "has_citations": False,
            "valid_citations": False,
            "citation_precision": False,
        }
    for idx in citations:
        if idx < 0 or idx >= len(chunks):
            return {
                "has_citations": True,
                "valid_citations": False,
                "citation_precision": False,
            }
    citation_supports: dict[int, bool] = {
        idx: chunk_supports_answer(answer=answer, chunk=chunks[idx])
        for idx in citations
    }
    citation_precision = sum(
        [sup for sup in citation_supports.values() if sup == True]
    ) / len(citations)

    if citation_precision < 1.0:
        print("\n[CITATION PRECISION FAILURE]")
        print(f"\nAnswer: {answer}")
        print(f"Citation precision: {citation_precision:.2f}")

        for idx, supported in citation_supports.items():
            status = "SUPPORTED" if supported else "NOT SUPPORTED"
            print(f"[{idx}] {status}")
            print(chunks[idx])

    return {
        "has_citations": True,
        "valid_citations": True,
        "citation_precision": citation_precision,
    }


def evaluate_claim_attribution(
    answer: str,
    chunks: list[str],
    support_fn: Callable[[str, str], bool] | None = None,
):
    """
    Claim-level citation evaluation.

    Splits the answer into sentence-level claims (via extract_cited_claims)
    and evaluates attribution per claim, rather than pooling all citations at
    the answer level as evaluate_citation_precision does. This exposes two
    failure modes the answer-level metric cannot:
    - a claim that cites the WRONG chunk (caught by claim precision), and
    - a claim asserted with NO citation at all (caught by claim coverage).

    A claim is supported if EVERY chunk it cites supports it. where "support"
    is decided by support_fn (default: lexical chunk_supports_answer). Pass
    a differente support_fn (e.g. an LLM-judged check) to strenghten it without
    changing this logic.

    claim_precision is None (not 0.0) when there are no cited claims, and
    claim_coverage is None when there are no claims, so empty or citation-free
    answers are excluded from aggregation rather than dragging averages to 0.
    """
    if support_fn is None:
        support_fn = chunk_supports_answer

    cited_claims_detail: list[dict[str, Any]] = []
    claims = extract_cited_claims(answer=answer)
    total_claims = len(claims)
    cited_claims = 0
    supported_claims = 0
    for claim in claims:
        claim_text = claim["claim"]
        indices = claim["citation_indices"]

        if not indices:
            cited_claims_detail.append(
                {
                    "claim": claim_text,
                    "citation_indices": [],
                    "cited": False,
                    "supported": None,
                }
            )
            continue

        cited_claims += 1

        valid = all(0 <= idx < len(chunks) for idx in indices)
        supported = valid and all(
            support_fn(claim_text, chunks[idx]) for idx in indices
        )
        if supported:
            supported_claims += 1

        cited_claims_detail.append(
            {
                "claim": claim_text,
                "citation_indices": indices,
                "cited": True,
                "supported": supported,
            }
        )
    claim_precision = supported_claims / cited_claims if cited_claims else None
    claim_coverage = cited_claims / total_claims if total_claims else None

    return {
        "total_claims": total_claims,
        "cited_claims": cited_claims,
        "supported_claims": supported_claims,
        "claim_precision": claim_precision,
        "claim_coverage": claim_coverage,
        "claims": cited_claims_detail,
    }
