# src/techniques/technique_extractor.py

from typing import Dict, List

from .phrase_miner import extract_candidate_phrases
from .filters import filter_phrases


class TechniqueExtractor:
    """
    Deterministic technique extractor for research abstracts.
    """

    def extract(self, abstract: str) -> Dict:
        raw_phrases = extract_candidate_phrases(abstract)
        techniques = filter_phrases(raw_phrases)

        return {
            "techniques": [
                {
                    "name": t,
                    "confidence": self._confidence(t)
                }
                for t in techniques
            ]
        }

    @staticmethod
    def _confidence(phrase: str) -> float:
        """
        Simple deterministic confidence score.
        """
        length_bonus = min(len(phrase.split()) / 5, 1.0)
        return round(0.6 + 0.4 * length_bonus, 3)