# src/techniques/phrase_miner.py

import re
from typing import List


METHOD_VERBS = {
    "propose", "introduce", "present", "develop",
    "design", "construct", "formulate", "study"
}


def extract_candidate_phrases(text: str) -> List[str]:
    """
    Extracts method-bearing noun phrases using rule patterns.
    Deterministic and fast.
    """
    sentences = re.split(r"[.!?]", text.lower())
    phrases = []

    for sent in sentences:
        if not any(v in sent for v in METHOD_VERBS):
            continue

        # Pattern: adjective* noun+ (simple but effective)
        matches = re.findall(
            r"(?:[a-z]+\s){0,3}(?:algorithm|method|construction|model|framework|protocol|spanner)",
            sent
        )

        for m in matches:
            phrases.append(m.strip())

    return list(set(phrases))