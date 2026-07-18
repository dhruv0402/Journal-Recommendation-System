from typing import Dict, Any, List

from src.detection.exact_match import exact_match, normalized_match
from src.detection.normalize import normalize_title
from src.detection.semantic_match import SemanticMatcher
from src.core.journal_aggregation import aggregate_journals


# ============================
# Confidence helpers
# ============================

def get_confidence_label(hybrid_score: float) -> str:
    if hybrid_score >= 0.75:
        return "Strong match"
    elif hybrid_score >= 0.55:
        return "Probable match"
    elif hybrid_score >= 0.35:
        return "Weak match"
    else:
        return "Low relevance"


def downgrade_confidence(confidence: str) -> str:
    mapping = {
        "Strong match": "Probable match",
        "Probable match": "Weak match",
        "Weak match": "Low relevance",
        "Low relevance": "Low relevance",
    }
    return mapping.get(confidence, confidence)


# ============================
# Adaptive weight logic
# ============================

def get_length_based_weights(query_len: int, article_len: int) -> tuple[float, float]:
    """
    Compute semantic vs overlap weights based on token-length difference.
    """
    diff = abs(query_len - article_len)

    if diff <= 2:
        return 0.5, 0.5
    elif diff <= 5:
        return 0.65, 0.35
    else:
        return 0.8, 0.2


# ============================
# Core Detection
# ============================

def detect_journal(input_title: str, df) -> Dict[str, Any]:

    # ----------------------------
    # Exact + Normalized matching
    # ----------------------------
    exact_matches = exact_match(input_title, df)
    normalized_matches = normalized_match(input_title, df)

    exact_match_flag = len(exact_matches) > 0
    normalized_match_flag = len(normalized_matches) > 0

    # ----------------------------
    # Normalize query once
    # ----------------------------
    normalized_query = normalize_title(input_title)
    query_token_len = len(normalized_query.split())

    # ----------------------------
    # Semantic matching
    # ----------------------------
    semantic_results = SemanticMatcher().find_similar(input_title, df)

    enriched_semantic_matches: List[Dict[str, Any]] = []

    for match in semantic_results:
        article_title = match["article_title"]
        article_len = len(normalize_title(article_title).split())

        # Length-difference adaptive weights
        semantic_weight, overlap_weight = get_length_based_weights(
            query_token_len,
            article_len
        )

        hybrid_score = (
            semantic_weight * match["semantic_score"]
            + overlap_weight * match["overlap_score"]
        )

        confidence = get_confidence_label(hybrid_score)

        # Semantic drift suppression
        if match["semantic_score"] >= 0.6 and match["overlap_score"] == 0:
            confidence = downgrade_confidence(confidence)

        enriched_semantic_matches.append({
            **match,
            "semantic_weight": semantic_weight,
            "overlap_weight": overlap_weight,
            "hybrid_score": round(hybrid_score, 4),
            "confidence": confidence,
        })

    # ----------------------------
    # Rank & trim
    # ----------------------------
    enriched_semantic_matches.sort(
        key=lambda x: x["hybrid_score"],
        reverse=True
    )

    TOP_K = 5
    enriched_semantic_matches = enriched_semantic_matches[:TOP_K]

    # ----------------------------
    # Journal aggregation
    # ----------------------------
    journal_predictions = aggregate_journals(
        semantic_matches=enriched_semantic_matches,
        df=df,
        top_n=5,
    )

    # ----------------------------
    # Final output
    # ----------------------------
    return {
        "exact_match": exact_match_flag,
        "normalized_match": normalized_match_flag,
        "exact_matches": exact_matches,
        "normalized_matches": normalized_matches,
        "semantic_matches": enriched_semantic_matches,
        "journal_predictions": journal_predictions,
    }