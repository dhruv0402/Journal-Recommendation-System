import numpy as np
from unittest.mock import MagicMock

from src.phase2.semantic_validator import SemanticValidator


def test_semantic_validator_similarity_and_confidence():
    validator = SemanticValidator()

    # -------- MOCK EMBEDDINGS --------
    fake_user_vec = np.array([1.0, 0.0])
    fake_ref_vec = np.array([0.9, 0.1])

    validator.embedder.embed = MagicMock(side_effect=[
        fake_user_vec,
        fake_ref_vec
    ])

    validator.embedder.cosine_similarity = MagicMock(return_value=0.82)

    # -------- MOCK TOPIC VALIDATION --------
    validator.topic_validator.validate = MagicMock(return_value={
        "heading": "Computer Networks",
        "alignment_score": 0.75
    })

    # -------- MOCK CONFIDENCE --------
    validator.confidence_scorer.score = MagicMock(return_value={
        "score": 0.80,
        "label": "HIGH",
        "explanation": "Strong semantic and topic alignment"
    })

    # -------- EXECUTE --------
    result = validator.validate(
        "Journal about computer networks",
        "Computer networks routing and protocols"
    )

    # -------- ASSERT --------
    assert result["embedding_similarity"] == 0.82
    assert result["topic_heading"] == "Computer Networks"
    assert result["topic_alignment"] == 0.75
    assert result["confidence"]["score"] == 0.80