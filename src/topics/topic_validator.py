from collections import Counter
from typing import Dict, List
import re


class TopicValidator:
    """
    Rule-based topic modeling for journal heading validation.
    """

    def __init__(self):
        self.topic_heading_map = {
            "networking": "Computer Networks & Communications",
            "quantum": "Quantum Information & Communication",
            "biology": "Computational Biology",
            "cyber-physical": "Cyber-Physical Systems",
            "security": "Network Security",
            "control": "Control Systems Engineering",
        }

    def _extract_topics(self, text: str) -> Counter:
        tokens = re.findall(r"[a-z]{4,}", text.lower())

        topic_keywords = {
            "networking": ["network", "routing", "latency", "bandwidth"],
            "quantum": ["quantum", "qubit", "entanglement", "decoherence"],
            "biology": ["protein", "cell", "biological", "pathway"],
            "cyber-physical": ["sensor", "physical", "actuator", "embedded"],
            "security": ["attack", "secure", "encryption"],
            "control": ["control", "feedback", "stability"],
        }

        scores = Counter()

        for topic, keys in topic_keywords.items():
            scores[topic] = sum(1 for t in tokens if t in keys)

        return scores

    def validate(self, abstract: str) -> Dict:
        """
        Returns topic alignment and recommended heading.
        """
        topic_scores = self._extract_topics(abstract)

        if not topic_scores:
            return {
                "heading": None,
                "alignment_score": 0.0
            }

        best_topic, score = topic_scores.most_common(1)[0]
        total = sum(topic_scores.values()) + 1e-6

        return {
            "heading": self.topic_heading_map.get(best_topic),
            "alignment_score": round(score / total, 3)
        }