# src/embedding/embedding_cache.py

import hashlib
import pickle
from pathlib import Path
from typing import Optional
import numpy as np


class EmbeddingCache:
    def __init__(self, cache_dir: str = ".embedding_cache"):
        self.cache_path = Path(cache_dir)
        self.cache_path.mkdir(parents=True, exist_ok=True)

    def _hash_text(self, text: str) -> str:
        normalized = " ".join(text.lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[np.ndarray]:
        key = self._hash_text(text)
        file_path = self.cache_path / f"{key}.pkl"

        if not file_path.exists():
            return None

        with open(file_path, "rb") as f:
            return pickle.load(f)

    def set(self, text: str, embedding: np.ndarray):
        key = self._hash_text(text)
        file_path = self.cache_path / f"{key}.pkl"

        with open(file_path, "wb") as f:
            pickle.dump(embedding, f)