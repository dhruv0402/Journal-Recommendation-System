from collections import Counter
from unittest.mock import MagicMock
from src.topics.topic_validator import TopicValidator


def test_topic_validator_alignment():
    validator = TopicValidator()

    validator._extract_topics = MagicMock(
        return_value=Counter({
            "networking": 4,
            "quantum": 1
        })
    )

    result = validator.validate("dummy text")

    assert result["heading"] == "Computer Networks & Communications"
    assert result["alignment_score"] > 0.0