import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.phase2.abstract_aggregation import aggregate_abstract_results
from src.phase2.final_decision import make_final_decision
from src.phase2.journal_heading_recommender import recommend_journal_headings
from src.phase2.scope_reranker import rerank_with_scope
from src.phase2.learning_reranker import rerank_with_learning, load_model

# load learning reranker model once at startup
load_model()

import os
import pickle
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

INDEX_PATH = os.path.join(BASE_DIR, "data", "faiss_index.bin")
META_PATH = os.path.join(BASE_DIR, "data", "faiss_meta.pkl")

# ---------------- GLOBALS ----------------
_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- DOMAIN FILTER ----------------

from src.journal_domain_map import journal_domain_map

# --- manual corrections (high confidence fixes) ---
DOMAIN_OVERRIDES = {
    "Computer Networks": "networks",
    "AI Open": "AI",
    "Journal of Systems and Software": "software",
    "Artificial Intelligence": "AI",
    "Advances in Engineering Software": "software",
}

# apply overrides
journal_domain_map.update(DOMAIN_OVERRIDES)

DOMAIN_TEXTS = {
    "systems": "distributed systems parallel computing scheduling architecture",
    "networks": "computer networks routing communication protocols wireless",
    "software": "software engineering testing development system design",
}

# precompute once
DOMAIN_EMBS = {
    k: _model.encode(v, normalize_embeddings=True) for k, v in DOMAIN_TEXTS.items()
}


def detect_domain(user_embedding):
    best_domain = None
    best_score = -1

    for domain, emb in DOMAIN_EMBS.items():
        score = float(np.dot(user_embedding[0], emb))
        if score > best_score:
            best_score = score
            best_domain = domain

    return best_domain


_index = None
_metadata = None

_journal_counts = None

_FAISS_INDEX = None
_DATASET_EMBEDDINGS = None

DUPLICATE_THRESHOLD = 0.95
NEAR_DUPLICATE_THRESHOLD = 0.80
# ---------------- BACKWARD COMPATIBILITY FIX ----------------
# Ensure API import works


def preload_dataset_embeddings(dataset_abstracts: list[str]):
    global _FAISS_INDEX, _DATASET_EMBEDDINGS

    PHASE0_INDEX_PATH = os.path.join(BASE_DIR, "data", "phase0_faiss.index")
    PHASE0_EMB_PATH = os.path.join(BASE_DIR, "data", "phase0_embeddings.pkl")

    # ---------- LOAD IF EXISTS ----------
    if os.path.exists(PHASE0_INDEX_PATH) and os.path.exists(PHASE0_EMB_PATH):
        print("[Phase0] Loading FAISS index...")

        _FAISS_INDEX = faiss.read_index(PHASE0_INDEX_PATH)

        with open(PHASE0_EMB_PATH, "rb") as f:
            _DATASET_EMBEDDINGS = pickle.load(f)

        print("[Phase0] Loaded.")
        return

    # ---------- BUILD ----------
    print("[Phase0] Building FAISS index...")

    embeddings = _model.encode(
        dataset_abstracts,
        normalize_embeddings=True,
        batch_size=256,
        show_progress_bar=True,
    )

    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    _FAISS_INDEX = index
    _DATASET_EMBEDDINGS = embeddings

    # ---------- SAVE ----------
    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

    faiss.write_index(_FAISS_INDEX, PHASE0_INDEX_PATH)

    with open(PHASE0_EMB_PATH, "wb") as f:
        pickle.dump(_DATASET_EMBEDDINGS, f)

    print("[Phase0] Built and saved.")


# ---------------- PRELOAD ----------------
def preload_phase2(df):
    """
    Run ONCE at startup
    Loads FAISS index if available, otherwise builds and saves it
    ALSO computes journal frequency for bias correction
    """
    global _index, _metadata, _journal_counts

    # ---------- LOAD IF EXISTS ----------
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        print("[Phase2] Loading FAISS index from disk...")

        _index = faiss.read_index(INDEX_PATH)

        with open(META_PATH, "rb") as f:
            _metadata = pickle.load(f)

        print("[Phase2] FAISS loaded.")

        # rebuild journal counts (cheap)
        _journal_counts = defaultdict(int)
        for _, journal in _metadata:
            _journal_counts[journal] += 1

        return

    # ---------- BUILD ----------
    print("[Phase2] Building FAISS index...")

    abstracts = df["abstract"].fillna("").tolist()
    journals = df["journal_name"].tolist()

    embeddings = _model.encode(
        abstracts,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=True,
    )

    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(np.array(embeddings))

    _index = index
    _metadata = list(zip(abstracts, journals))

    # ---------- SAVE ----------
    os.makedirs("data", exist_ok=True)
    faiss.write_index(_index, INDEX_PATH)

    with open(META_PATH, "wb") as f:
        pickle.dump(_metadata, f)

    print("[Phase2] FAISS built and saved.")

    # ---------- JOURNAL COUNTS ----------
    _journal_counts = defaultdict(int)
    for journal in journals:
        _journal_counts[journal] += 1


# ---------------- MAIN ENGINE ----------------
def run_phase2_fast(user_abstract: str, top_k: int = 20, training_mode: bool = False):
    """
    Fast Phase 2 using FAISS retrieval + bias correction
    """

    global _index, _metadata, _journal_counts

    if _index is None:
        raise Exception("Phase2 not initialized. Call preload_phase2() first")

    # Encode user abstract
    user_embedding = _model.encode([user_abstract], normalize_embeddings=True)

    # -------- DOMAIN DETECTION --------
    user_domain = detect_domain(user_embedding)

    # FAISS search
    scores, indices = _index.search(user_embedding, top_k)

    article_results = []

    for score, idx in zip(scores[0], indices[0]):
        abstract, journal = _metadata[idx]

        # ---------------- SCORE BOOST ----------------
        boosted_score = float(score)

        # ---------------- BIAS CORRECTION ----------------
        journal_freq = _journal_counts[journal]
        bias_penalty = 1 / (1 + 0.1 * np.log1p(journal_freq))  # very light penalty

        final_score = boosted_score * bias_penalty

        article_results.append(
            {
                "journal_name": journal,
                "similarity": float(final_score),
                "abstract": abstract,
            }
        )

    # -------- DOMAIN WEIGHTING (soft filter) --------
    for r in article_results:
        domain = journal_domain_map.get(r["journal_name"])

        # handle missing domain safely (no penalty if unknown)
        if domain is None:
            continue

        if domain == user_domain:
            r["similarity"] *= 1.15  # boost matching domain
        else:
            r["similarity"] *= 0.90  # mild penalty only

    # ---------------- AGGREGATION ----------------
    journal_predictions = aggregate_abstract_results(article_results)

    # -------- TRAINING MODE SHORT-CIRCUIT --------
    if training_mode:
        return {"journal_predictions": journal_predictions}

    # -------- RERANK WITH JOURNAL SCOPE --------
    journal_predictions = rerank_with_scope(user_embedding, journal_predictions)

    # -------- LEARNING-BASED RERANK --------
    journal_predictions = rerank_with_learning(journal_predictions)
    print("Before reranker:", journal_predictions[:3])
    journal_predictions = rerank_with_learning(journal_predictions)
    print("After reranker:", journal_predictions[:3])
    # ---------------- NORMALIZE CONFIDENCE (FIX SATURATION) ----------------
    if journal_predictions:
        # Max-normalization to avoid hard 1.0 saturation
        max_conf = max(j.get("confidence", 0.0) for j in journal_predictions) or 1.0

        for j in journal_predictions:
            j["confidence"] = round(j.get("confidence", 0.0) / max_conf, 3)

    # ---------------- FINAL DECISION ----------------
    submission = make_final_decision(journal_predictions)

    # -------- HARD SAFETY FIX --------
    # Some paths still return string → normalize EVERYTHING
    if isinstance(submission, str):
        submission = {"journal": submission, "confidence": 0.0}

    if not isinstance(submission, dict):
        submission = {"journal": str(submission), "confidence": 0.0}

    submission.setdefault("journal", "Unknown")
    submission.setdefault("confidence", 0.0)

    if isinstance(submission, str):
        submission = {"journal": submission, "confidence": 0.0}

    confidence = submission.get("confidence", 0.0)

    if confidence >= 0.60:
        final_decision = "Strong journal scope match"
    elif confidence >= 0.30:
        final_decision = "Partial journal scope match"
    else:
        final_decision = "Novel article – no strong journal scope match"
    # ---------------- HEADINGS ----------------
    recommended_headings = None
    if confidence >= 0.30:
        recommended_headings = recommend_journal_headings({}, top_k=3)

    best_journal = submission.get("journal", "No suitable journal")

    if confidence >= 0.30:
        final_recommendation = f"Submit to {best_journal}"
    elif confidence >= 0.20:
        final_recommendation = f"Possible fit: {best_journal}"
    else:
        final_recommendation = "No suitable journal found"

    return {
        "journal_predictions": journal_predictions,
        "final_decision": final_decision,
        "recommended_headings": recommended_headings,
        "submission_recommendation": submission,
        "final_recommendation": final_recommendation,
        "best_journal": best_journal,
        "best_confidence": confidence,
    }


# ---------------- BACKWARD COMPATIBILITY FIX ----------------
# Ensure API import works


def check_against_dataset(user_abstract: str, dataset_abstracts: list[str]):
    """
    Wrapper for API compatibility (Phase0)
    """
    global _FAISS_INDEX

    if _FAISS_INDEX is None:
        preload_dataset_embeddings(dataset_abstracts)

    user_emb = _model.encode(user_abstract, normalize_embeddings=True).astype("float32")

    scores, _ = _FAISS_INDEX.search(np.array([user_emb]), 50)

    max_similarity = float(scores[0][0])

    if max_similarity >= DUPLICATE_THRESHOLD:
        verdict = "DUPLICATE"
    elif max_similarity >= NEAR_DUPLICATE_THRESHOLD:
        verdict = "NEAR_DUPLICATE"
    else:
        verdict = "DISTINCT"

    return {"verdict": verdict, "confidence": round(max_similarity, 3)}
