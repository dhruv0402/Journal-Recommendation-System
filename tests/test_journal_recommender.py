from src.phase2.journal_recommender import recommend_submission_journals


def test_clear_primary():
    preds = [
        {"journal_name": "A", "avg_top_similarity": 0.85, "max_similarity": 0.90, "article_matches": 5},
        {"journal_name": "B", "avg_top_similarity": 0.72, "max_similarity": 0.75, "article_matches": 3},
    ]
    result = recommend_submission_journals(preds)
    assert result["primary_journal"] == "A"
    assert not result["alternate_journals"]


def test_multiple_close():
    preds = [
        {"journal_name": "A", "avg_top_similarity": 0.81, "max_similarity": 0.85, "article_matches": 4},
        {"journal_name": "B", "avg_top_similarity": 0.79, "max_similarity": 0.82, "article_matches": 4},
        {"journal_name": "C", "avg_top_similarity": 0.78, "max_similarity": 0.80, "article_matches": 3},
    ]
    result = recommend_submission_journals(preds)
    assert result["primary_journal"] == "A"
    assert len(result["alternate_journals"]) > 0


def test_no_predictions():
    result = recommend_submission_journals([])
    assert result is None
