from typing import Set, Tuple, List
from src.detection.normalize import normalize_title


def get_tokens(text: str) -> Set[str]:
    if not text:
        return set()

    normalized = normalize_title(text)
    return set(normalized.split())


def get_bigrams(tokens: List[str]) -> Set[Tuple[str, str]]:
    if len(tokens) < 2:
        return set()

    return {
        (tokens[i], tokens[i + 1])
        for i in range(len(tokens) - 1)
    }


def token_overlap_score(a: str, b: str) -> float:
    tokens_a = get_tokens(a)
    tokens_b = get_tokens(b)

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = tokens_a.intersection(tokens_b)
    union = tokens_a.union(tokens_b)

    return len(intersection) / len(union)


def bigram_overlap_score(a: str, b: str) -> float:
    tokens_a = list(get_tokens(a))
    tokens_b = list(get_tokens(b))

    bigrams_a = get_bigrams(tokens_a)
    bigrams_b = get_bigrams(tokens_b)

    if not bigrams_a or not bigrams_b:
        return 0.0

    intersection = bigrams_a.intersection(bigrams_b)
    union = bigrams_a.union(bigrams_b)

    return len(intersection) / len(union)
def compute_overlap_score(a: str, b: str) -> dict:
    """
    Compute combined lexical overlap score using token and bigram overlap.

    Args:
        a (str): First text
        b (str): Second text

    Returns:
        dict: Overlap metrics including token, bigram, and combined score
    """
    token_overlap = token_overlap_score(a, b)
    bigram_overlap = bigram_overlap_score(a, b)

    overlap_score = (0.6 * token_overlap) + (0.4 * bigram_overlap)

    return {
        "token_overlap": token_overlap,
        "bigram_overlap": bigram_overlap,
        "overlap_score": overlap_score,
    }