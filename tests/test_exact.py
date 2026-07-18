import pandas as pd
from src.detection.exact_match import exact_and_normalized_match

def test_exact_match_found():
    data = pd.DataFrame({
        "article_title": ["Journal of Computer Networks"]
    })

    result = exact_and_normalized_match(
        "Journal of Computer Networks",
        data
    )

    assert result["exact_match"] is True
    assert "Journal of Computer Networks" in result["exact_matches"]


def test_normalized_match_found():
    data = pd.DataFrame({
        "article_title": ["Journal of Computer Networks"]
    })

    result = exact_and_normalized_match(
        "journal of computer networks",
        data
    )

    assert result["normalized_match"] is True