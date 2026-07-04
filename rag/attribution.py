import re
from rag.helpers import normalize
from typing import Any, Dict, List, Set, TypedDict
from nltk import sent_tokenize

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
    citation_indices: List[int]


def remove_citations(text: str) -> str:
    """Removes numeric citations such as [0] and [12] from text."""
    text_without_citations = re.sub(r"\[\d+\]", "", text)

    # Remove spaces left before punctuation: "claim ." -> "claim."
    text_without_citations = re.sub(r"\s+([.,;:!?])", r"\1", text_without_citations)
    # Normalize repeated whitespace
    return re.sub(r"\s+", " ", text_without_citations).strip()


def extract_cited_claims(answer: str) -> List[CitedClaim]:
    """
    Split an answer into sentence level claims and extracts the citations
    attached to each claim.

    Sentences without citations are retained with an empty citation list.
    This allows citation completeness to be evaluated later.
    """
    if not answer.strip():
        return []

    claims: List[CitedClaim] = []

    for sentence in sent_tokenize(answer):
        citation_indices = list(dict.fromkeys(extract_citations(answer=sentence)))
        claim_text = remove_citations(text=sentence)
        if not claim_text:
            continue

        claims.append({"claim": claim_text, "citation_indices": citation_indices})

    return claims


def extract_citations(answer: str) -> List[int]:
    """
    Extract citation indices from answer.

    Supports:
    [1], [2]
    (1), (2)
    """
    bracket_matches: list = re.findall(r"\[(\d+)\]", answer)
    paren_matches: list = re.findall(r"\((\d+)\)", answer)

    citations = bracket_matches + paren_matches

    return [int(c) for c in citations]


def strip_citations(text: str) -> str:
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\(\d+\)", "", text)
    return text


def chunk_supports_answer(answer: str, chunk: str) -> bool:
    clean_answer: str = strip_citations(text=answer)

    answer_words: Set[str] = {
        w for w in normalize(text=clean_answer).split() if w not in STOPWORDS
    }
    chunk_words: Set[str] = {
        w for w in normalize(text=chunk).split() if w not in STOPWORDS
    }

    overlap = len(answer_words & chunk_words)
    coverage = overlap / max(len(answer_words), 1)

    return coverage >= 0.5


def evaluate_faithfulness(answer: str, chunks: List[str]) -> Dict[str, bool]:
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


def evaluate_citation_precision(answer: str, chunks: List[str]) -> Dict[str, Any]:
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
    citations: List[int] = list(dict.fromkeys(extract_citations(answer=answer)))

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
    citation_supports = {
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
