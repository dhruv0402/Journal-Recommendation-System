from collections import defaultdict
from typing import List, Dict, Any


def journal_confidence_label(max_score: float) -> str:
    if max_score >= 0.75:
        return "Strong journal match"
    elif max_score >= 0.55:
        return "Probable journal match"
    elif max_score >= 0.35:
        return "Weak journal match"
    else:
        return "Low relevance"


def aggregate_journals(
    semantic_matches: List[Dict[str, Any]],
    df,
    top_n: int = 5,
) -> List[Dict[str, Any]]:

    journal_stats = defaultdict(lambda: {
        "scores": [],
        "count": 0,
    })

    # Map article → journal
    title_to_journal = dict(
        zip(df["article_title"], df["journal_name"])
    )

    for match in semantic_matches:
        article_title = match["article_title"]
        hybrid_score = match["hybrid_score"]

        journal = title_to_journal.get(article_title)
        if not journal:
            continue

        journal_stats[journal]["scores"].append(hybrid_score)
        journal_stats[journal]["count"] += 1

    journal_predictions = []

    for journal, stats in journal_stats.items():
        avg_score = sum(stats["scores"]) / len(stats["scores"])
        max_score = max(stats["scores"])

        journal_predictions.append({
            "journal_name": journal,
            "article_matches": stats["count"],
            "avg_hybrid_score": round(avg_score, 4),
            "max_hybrid_score": round(max_score, 4),
            "confidence": journal_confidence_label(max_score),
        })

    # Rank journals by max evidence
    journal_predictions.sort(
        key=lambda x: x["max_hybrid_score"],
        reverse=True
    )

    return journal_predictions[:top_n]