# src/techniques/filters.py

GENERIC_PHRASES = {
    "performance evaluation",
    "simulation results",
    "experimental analysis",
    "numerical results",
}


def filter_phrases(phrases):
    filtered = []

    for p in phrases:
        if len(p.split()) < 2:
            continue
        if p in GENERIC_PHRASES:
            continue
        filtered.append(p)

    return filtered