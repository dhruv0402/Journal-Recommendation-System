import numpy as np
from src.phase2.journal_scope_map import journal_scope_map

# Lazily resolved — avoids loading a second copy of the SentenceTransformer model.
# optimized_phase2 already holds the singleton; we borrow it at first call.
_model = None
_SCOPE_EMBS = {}


def _ensure_loaded():
    global _model, _SCOPE_EMBS
    if _model is not None:
        return
    # Import here to avoid circular imports at module load time
    from src.phase2.optimized_phase2 import _model as shared_model
    _model = shared_model
    _SCOPE_EMBS = {
        j: _model.encode(text, normalize_embeddings=True)
        for j, text in journal_scope_map.items()
    }


def rerank_with_scope(user_embedding, journal_predictions):
    """
    Boost journal confidence using scope similarity.
    Reuses the shared SentenceTransformer instance from optimized_phase2.
    """
    _ensure_loaded()

    for j in journal_predictions:
        journal = j.get("journal_name")

        if journal not in _SCOPE_EMBS:
            continue

        scope_emb = _SCOPE_EMBS[journal]

        # cosine similarity
        sim = float(np.dot(user_embedding[0], scope_emb))

        # 🔥 improved boost with stability + clamp
        base_conf = j.get("confidence", 0.0)

        # weighted combination (slightly stronger semantic influence)
        new_conf = base_conf * 0.65 + sim * 0.35

        # clamp to valid range [0, 1]
        new_conf = max(0.0, min(1.0, new_conf))

        j["confidence"] = round(new_conf, 3)

    # sort again after reranking
    journal_predictions.sort(key=lambda x: x["confidence"], reverse=True)

    return journal_predictions
