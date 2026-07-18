from collections import defaultdict
import math

JOURNAL_CORPUS_SIZES = {
    "Engineering Applications of Artificial Intelligence": 3910,
    "Computer Networks": 714,
    "Journal of Systems and Software": 646,
    "Computer Vision and Image Understanding": 634,
    "Advances in Engineering Software": 352,
    "Computer Communications": 232,
    "Artificial Intelligence": 218,
    "Parallel Computing": 160,
    "Computer Standards & Interfaces": 125,
    "AI Open": 100,
    "Journal of Computer and System Sciences": 44,
}

_MEDIAN_CORPUS_SIZE = 232


def _corpus_penalty(journal_name: str) -> float:
    size = JOURNAL_CORPUS_SIZES.get(journal_name, _MEDIAN_CORPUS_SIZE)
    if size <= _MEDIAN_CORPUS_SIZE:
        return 1.0  # no penalty for at-or-below median journals
    ratio = size / _MEDIAN_CORPUS_SIZE
    penalty = 1.0 / math.log2(ratio + 1)
    return round(min(1.0, penalty), 4)


def aggregate_abstract_results(article_results, top_k_per_journal=5):
    journal_scores = defaultdict(list)

    for r in article_results:
        journal_scores[r["journal_name"]].append(r["similarity"])

    aggregated = []

    for journal, scores in journal_scores.items():
        scores = sorted(scores, reverse=True)
        top_scores = scores[:top_k_per_journal]

        weighted_score = sum(s * (1 / (i + 1)) for i, s in enumerate(top_scores))
        weight_norm = sum(1 / (i + 1) for i in range(len(top_scores)))
        scaled_score = weighted_score / (1 + 0.3 * (len(top_scores) - 1))
        normalized_score = scaled_score / weight_norm if weight_norm > 0 else 0

        penalty = _corpus_penalty(journal)
        normalized_score = normalized_score * penalty
        normalized_score = min(0.85, normalized_score)

        aggregated.append(
            {
                "journal_name": journal,
                "confidence": round(float(normalized_score), 3),
                "similarity": round(float(sum(top_scores) / len(top_scores)), 3)
                if top_scores
                else 0.0,
                "top_match": round(float(top_scores[0]), 3) if top_scores else 0.0,
                "matches_used": len(top_scores),
                "corpus_penalty": penalty,
            }
        )

    return sorted(aggregated, key=lambda x: x["confidence"], reverse=True)
