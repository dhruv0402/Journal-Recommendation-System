from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

from .embedding_cache import EmbeddingCache


class EmbeddingEngine:
    """
    Deterministic, cached abstract embedding engine.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: str = ".embedding_cache"
    ):
        self.model = SentenceTransformer(model_name)
        self.cache = EmbeddingCache(cache_dir)

    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single abstract (cached).
        """
        cached = self.cache.get(text)
        if cached is not None:
            return cached

        embedding = self.model.encode(
            text,
            normalize_embeddings=True
        )

        self.cache.set(text, embedding)
        return embedding

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embed multiple abstracts efficiently.
        """
        embeddings = [None] * len(texts)
        to_compute = []
        index_map = []

        for idx, text in enumerate(texts):
            cached = self.cache.get(text)
            if cached is not None:
                embeddings[idx] = cached
            else:
                to_compute.append(text)
                index_map.append(idx)

        if to_compute:
            computed = self.model.encode(
                to_compute,
                normalize_embeddings=True
            )

            for idx, emb in zip(index_map, computed):
                self.cache.set(texts[idx], emb)
                embeddings[idx] = emb

        return np.vstack(embeddings)

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        # embeddings are normalized → dot product == cosine similarity
        return float(np.dot(vec1, vec2))