from src.core.pipeline import run_pipeline


def test_pipeline_runs_without_detection_logic():
    """
    End-to-end sanity test.

    This test ensures:
    - Pipeline executes
    - Contract shape is preserved
    - No detection logic is required yet
    """

    result = run_pipeline("Dummy Journal Title")

    assert isinstance(result, dict)

    assert "exact_match" in result
    assert "normalized_match" in result
    assert "exact_matches" in result
    assert "semantic_matches" in result

    assert isinstance(result["exact_match"], bool)
    assert isinstance(result["normalized_match"], bool)
    assert isinstance(result["exact_matches"], list)
    assert isinstance(result["semantic_matches"], list)
