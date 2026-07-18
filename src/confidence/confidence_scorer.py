from typing import Dict


class ConfidenceScorer:
    """
    Combines embedding similarity and topic alignment
    into a final confidence score with explanation.
    """

    def __init__(
        self,
        embedding_weight: float = 0.7,
        topic_weight: float = 0.3
    ):
        self.embedding_weight = embedding_weight
        self.topic_weight = topic_weight

    def score(
        self,
        embedding_similarity: float,
        topic_alignment: float
    ) -> Dict:

        # ---------- WEIGHTED SCORE ----------
        final_score = (
            self.embedding_weight * embedding_similarity +
            self.topic_weight * topic_alignment
        )

        final_score = round(final_score, 3)

        # ---------- VERDICT ----------
        if final_score >= 0.75:
            verdict = "HIGH"
        elif final_score >= 0.5:
            verdict = "MEDIUM"
        else:
            verdict = "LOW"

        # ---------- EXPLANATION ----------
        explanation = (
            f"Embedding similarity={embedding_similarity:.2f}, "
            f"topic alignment={topic_alignment:.2f}. "
            f"Overall confidence is {verdict}."
        )

        return {
            "score": final_score,
            "verdict": verdict,
            "explanation": explanation
        }