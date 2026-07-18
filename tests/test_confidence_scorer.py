from src.confidence.confidence_scorer import ConfidenceScorer


def test_high_confidence():
    scorer = ConfidenceScorer()

    result = scorer.score(
        embedding_similarity=0.85,
        topic_alignment=0.80
    )

    assert result["verdict"] == "HIGH"
    assert result["score"] >= 0.8


def test_medium_confidence():
    scorer = ConfidenceScorer()

    result = scorer.score(
        embedding_similarity=0.65,
        topic_alignment=0.40
    )

    assert result["verdict"] == "MEDIUM"


def test_low_confidence():
    scorer = ConfidenceScorer()

    result = scorer.score(
        embedding_similarity=0.35,
        topic_alignment=0.10
    )

    assert result["verdict"] == "LOW"