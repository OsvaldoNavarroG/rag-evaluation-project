import re
from evaluation import normalize
from typing import Dict, List


def extract_citations(answer: str) -> list:
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


def chunk_supports_answer(answer: str, chunk: str):
    answer_norm: str = normalize(text=answer)
    chunk_norm: str = normalize(text=chunk)

    return answer_norm in chunk_norm


def evaluate_faithfulness(answer: str, chunks: List[str]) -> Dict[str, bool]:
    """
    Checks whether cited chunks support the answer.

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
