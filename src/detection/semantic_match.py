from sentence_transformers import SentenceTransformer

from src.detection.similarity_utils import cosine_similarity
from src.detection.overlap_similarity import compute_overlap_score


class SemanticMatcher:
    def __init__(self, model_name="all-MiniLM-L6-v2", threshold=0.4):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold

    def find_similar(self, query: str, df):
        if not query or df is None or df.empty:
            return []

        titles = df["article_title"].dropna().astype(str).tolist()

        query_embedding = self.model.encode(query)
        corpus_embeddings = self.model.encode(titles)

        results = []

        for idx, title in enumerate(titles):
            semantic_score = cosine_similarity(
                query_embedding,
                corpus_embeddings[idx]
            )

            if semantic_score < self.threshold:
                continue

            overlap = compute_overlap_score(query, title)

            results.append({
                "article_title": title,
                "semantic_score": round(semantic_score, 4),
                "token_overlap": overlap["token_overlap"],
                "bigram_overlap": overlap["bigram_overlap"],
                "overlap_score": overlap["overlap_score"],
            })

        return results