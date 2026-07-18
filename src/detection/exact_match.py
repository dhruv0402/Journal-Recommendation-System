from src.detection.normalize import normalize_title


def exact_match(title: str, df):
    if not title:
        return []

    return df[df["article_title"] == title]["article_title"].tolist()


def normalized_match(title: str, df):
    if not title:
        return []

    normalized_input = normalize_title(title)

    df["_normalized"] = df["article_title"].apply(normalize_title)

    return df[df["_normalized"] == normalized_input]["article_title"].tolist()


def exact_and_normalized_match(title: str, df) -> dict:
    """Combined wrapper used by tests and the detector."""
    em = exact_match(title, df)
    nm = normalized_match(title, df)
    return {
        "exact_match": len(em) > 0,
        "normalized_match": len(nm) > 0,
        "exact_matches": em,
        "normalized_matches": nm,
    }