from rapidfuzz import process, fuzz

TITLE_DUP_THRESHOLD = 95.0
TITLE_NEAR_THRESHOLD = 85.0


def preload_title_embeddings(dataset_titles: list[str]):
    # Keep function for backward compatibility
    pass


def check_title_against_dataset(user_title: str, dataset_titles: list[str]):
    if not user_title or not dataset_titles:
        return {
            "status": "OK",
            "confidence": 0.0
        }

    # Use rapidfuzz process.extractOne with token_set_ratio to handle prefixes/acronyms correctly
    best_match = process.extractOne(
        user_title,
        dataset_titles,
        scorer=fuzz.token_set_ratio
    )

    if best_match:
        match_title, score, match_idx = best_match
        confidence = score / 100.0

        if score >= TITLE_DUP_THRESHOLD:
            verdict = "EXACT_MATCH"
        elif score >= TITLE_NEAR_THRESHOLD:
            verdict = "NEAR_MATCH"
        else:
            verdict = "OK"
    else:
        verdict = "OK"
        confidence = 0.0

    return {
        "status": verdict,
        "confidence": round(confidence, 3)
    }