from src.embeddings.embedding_engine import EmbeddingEngine
from src.topics.topic_validator import TopicValidator
from src.confidence.confidence_scorer import ConfidenceScorer
from src.techniques.technique_extractor import TechniqueExtractor


class SemanticValidator:
    """
    Independent semantic validation layer.
    """

    def __init__(self):
        self.embedder = EmbeddingEngine()
        self.topic_validator = TopicValidator()
        self.confidence_scorer = ConfidenceScorer()
        self.technique_extractor = TechniqueExtractor()  # 🔹 NEW

    def validate(
        self,
        user_abstract: str,
        reference_abstract: str
    ) -> dict:

        # -------- EMBEDDINGS --------
        u_vec = self.embedder.embed(user_abstract)
        r_vec = self.embedder.embed(reference_abstract)
        embedding_sim = self.embedder.cosine_similarity(u_vec, r_vec)

        # -------- TOPIC VALIDATION --------
        topic_result = self.topic_validator.validate(user_abstract)

        # -------- TECHNIQUE EXTRACTION --------
        techniques = self.technique_extractor.extract(user_abstract)

        # -------- CONFIDENCE --------
        confidence = self.confidence_scorer.score(
            embedding_similarity=embedding_sim,
            topic_alignment=topic_result["alignment_score"]
        )

        return {
    "embedding_similarity": round(embedding_sim, 3),
    "topic_heading": topic_result["heading"],
    "topic_alignment": topic_result["alignment_score"],
    "techniques": techniques,
    "confidence": confidence
}