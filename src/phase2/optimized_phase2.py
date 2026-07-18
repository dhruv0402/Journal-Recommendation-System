import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.phase2.abstract_aggregation import aggregate_abstract_results
from src.phase2.final_decision import make_final_decision
from src.phase2.journal_heading_recommender import recommend_journal_headings
from src.phase2.scope_reranker import rerank_with_scope
from src.phase2.learning_reranker import rerank_with_learning, load_model

import os
import pickle

from src.rag.rag_engine import RAGEngine

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

INDEX_PATH = os.path.join(BASE_DIR, "data", "faiss_index.bin")
META_PATH = os.path.join(BASE_DIR, "data", "faiss_meta.pkl")

# ---------------- GLOBALS ----------------
_model = SentenceTransformer("all-MiniLM-L6-v2")
_index = None
_metadata = None
_rag_engine = None
_query_cache = {}


# ---------------- PRELOAD ----------------
def preload_phase2(df, force_rebuild=False):
    global _index, _metadata

    # Load reranker model weights
    load_model()

    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH) and not force_rebuild:
        print("[Phase2] Loading FAISS index from disk...")
        _index = faiss.read_index(INDEX_PATH)

        with open(META_PATH, "rb") as f:
            _metadata = pickle.load(f)

        print("[Phase2] FAISS loaded.")
    else:
        print("[Phase2] Building FAISS index...")

        abstracts = df["abstract"].fillna("").tolist()
        journals = df["journal_name"].tolist()

        embeddings = _model.encode(
            abstracts,
            normalize_embeddings=True,
            batch_size=256,
            show_progress_bar=True,
        )

        dim = embeddings.shape[1]

        index = faiss.IndexFlatIP(dim)
        index.add(np.array(embeddings))

        _index = index
        _metadata = list(zip(abstracts, journals))

        if not force_rebuild:
            os.makedirs("data", exist_ok=True)
            faiss.write_index(_index, INDEX_PATH)

            with open(META_PATH, "wb") as f:
                pickle.dump(_metadata, f)

            print("[Phase2] FAISS built and saved.")
        else:
            print("[Phase2] FAISS built in-memory.")

    global _rag_engine
    if _rag_engine is None:
        try:
            print("[Phase2] Initializing RAG engine...")
            _rag_engine = RAGEngine(df)
        except Exception as e:
            print("[Phase2] RAG init failed:", e)


def run_phase2_fast(user_abstract: str, top_k: int = 50, training_mode: bool = False):
    """
    Fast Phase 2 using FAISS retrieval + LLM-as-validator pattern.

    Pipeline flow:
    1. FAISS retrieval → top-K similar abstracts
    2. Aggregation → per-journal confidence scores
    3. Scope reranker + learning reranker → refined ranking
    4. LLM validator → independently picks best journal from top-3 candidates
       (may disagree with pipeline — disagreement is logged and surfaced)
    5. Final decision merges pipeline confidence + LLM validation signal
    """

    global _index, _metadata

    if _index is None:
        raise Exception("Phase2 not initialized. Call preload_phase2() first")

    # -------- EMBEDDING CACHE --------
    if user_abstract in _query_cache:
        user_embedding = _query_cache[user_abstract]
    else:
        user_embedding = _model.encode([user_abstract], normalize_embeddings=True)
        _query_cache[user_abstract] = user_embedding

    # FAISS search
    k = min(top_k, 50)
    scores, indices = _index.search(user_embedding, k)

    article_results = []

    for score, idx in zip(scores[0], indices[0]):
        abstract, journal = _metadata[idx]
        article_results.append(
            {"journal_name": journal, "similarity": float(score), "abstract": abstract}
        )

    # ---------------- CLAMP RAW FAISS SCORES ----------------
    # Keep raw cosine similarity — do NOT min-max normalize.
    # Min-max normalization artificially inflates the top result to 1.0
    # regardless of actual semantic similarity, causing false confidence.
    for a in article_results:
        a["similarity"] = round(max(0.0, min(1.0, a["similarity"])), 3)

    # ---------------- AGGREGATION ----------------
    journal_predictions = aggregate_abstract_results(article_results)

    # -------- TRAINING MODE SHORT-CIRCUIT --------
    if training_mode:
        return {"journal_predictions": journal_predictions}

    # -------- APPLY RERANKERS --------
    journal_predictions = rerank_with_scope(user_embedding, journal_predictions)
    journal_predictions = rerank_with_learning(journal_predictions)

    # -------- CLAMP ALL PER-JOURNAL CONFIDENCE SCORES --------
    for j in journal_predictions:
        j["confidence"] = round(max(0.0, min(1.0, j.get("confidence", 0.0))), 3)
        j["similarity"] = round(max(0.0, min(1.0, j.get("similarity", 0.0))), 3)

    raw_scores = [j.get("confidence", 0.0) for j in journal_predictions]

    # ---------------- UNCERTAINTY SIGNALS ----------------
    spread = 0.0

    if raw_scores:
        sorted_scores = sorted(raw_scores, reverse=True)
        top3 = sorted_scores[:3]
        if len(top3) >= 2:
            spread = top3[0] - top3[-1]

    # ---------------- FINAL DECISION ----------------
    submission = make_final_decision(journal_predictions, abstract=user_abstract)

    if isinstance(submission, str):
        submission = {"journal": submission, "confidence": 0.0}

    # ---------------- COMPUTE FINAL CONFIDENCE ----------------
    if raw_scores:
        sorted_scores = sorted(raw_scores, reverse=True)

        top1 = sorted_scores[0]
        top2 = sorted_scores[1] if len(sorted_scores) > 1 else 0.0

        margin = max(0.0, top1 - top2)

        best_idx = int(np.argmax(raw_scores))
        best_similarity = journal_predictions[best_idx].get("similarity", 0.0)

        # Boosted similarity weight — raw semantic match drives confidence more
        confidence = 0.8 * best_similarity + 0.2 * (margin / (margin + 0.4))

        # Penalize weak semantic matches
        if best_similarity < 0.5:
            confidence *= 0.8
        if best_similarity < 0.3:
            confidence *= 0.6

        confidence = max(0.0, min(confidence, 1.0))
        confidence = round(confidence, 3)

    else:
        confidence = 0.0

    # -------- STABILITY FIX --------
    if len(journal_predictions) > 1:
        sorted_preds = sorted(
            journal_predictions, key=lambda x: x.get("confidence", 0.0), reverse=True
        )
        top1_conf = sorted_preds[0].get("confidence", 0.0)
        top2_conf = sorted_preds[1].get("confidence", 0.0)
        if abs(top1_conf - top2_conf) < 0.05:
            submission["journal"] = sorted_preds[0].get(
                "journal_name", submission.get("journal")
            )

    submission["confidence"] = confidence

    # ---------------- TOP-3 ----------------
    top3_recommendations = (
        sorted(
            journal_predictions,
            key=lambda x: (
                round(x.get("confidence", 0.0), 4),
                x.get("journal_name", ""),
            ),
            reverse=True,
        )[:3]
        if journal_predictions
        else []
    )

    # ---------------- RAG VALIDATION ----------------
    # LLM-as-validator: Groq independently evaluates all 3 candidates
    # and picks the best fit — it is NOT told which one the pipeline picked.
    # If it disagrees, that's a retrieval error signal logged and surfaced to UI.
    rag_explanations = {}

    if _rag_engine is not None and top3_recommendations:
        try:
            validated = _rag_engine.generate(
                user_abstract,
                top3_recommendations,
                extra_context={
                    "top_similarity": top3_recommendations[0].get("similarity", 0.0)
                },
            )

            pipeline_agreement = validated.get("pipeline_agreement", True)
            llm_pick = validated.get("best_journal", submission.get("journal"))

            # Log disagreements — these are the interesting cases for the paper
            if not pipeline_agreement:
                print(
                    f"[RAG OVERRIDE] Pipeline: {submission.get('journal')} → "
                    f"LLM pick: {llm_pick} | "
                    f"Reason: {validated.get('reason', '')[:100]}"
                )

            rag_explanations = {
                "global_explanation": validated,
                "pipeline_agreement": pipeline_agreement,
                "llm_pick": llm_pick,
                "override": validated.get("override", False),
            }

        except Exception as e:
            print(f"[RAG ERROR] {e}")
            rag_explanations = {"error": str(e)}

    # ---------------- FINAL DECISION THRESHOLDS ----------------
    # Thresholds lowered to match realistic cosine similarity ranges.
    # all-MiniLM-L6-v2 scores for good matches land in 0.55-0.70,
    # not 0.80+ like fine-tuned domain models.
    if confidence >= 0.50:
        final_decision = "Strong journal scope match"
    elif confidence >= 0.35:
        final_decision = "Partial journal scope match"
    else:
        final_decision = "Novel article – no strong journal scope match"

    recommended_headings = None
    if confidence >= 0.50:
        recommended_headings = recommend_journal_headings({}, top_k=3)

    best_journal = submission.get("journal", "No suitable journal")

    if confidence >= 0.50:
        final_recommendation = f"Submit to {best_journal}"
    elif confidence >= 0.35:
        final_recommendation = f"Possible fit: {best_journal}"
    else:
        final_recommendation = "No suitable journal found"

    # ---------------- LOGGING ----------------
    log_entry = {
        "input": user_abstract,
        "predicted_top1": best_journal,
        "confidence": confidence,
        "top3": [j.get("journal_name") for j in top3_recommendations],
        "llm_pick": rag_explanations.get("llm_pick", best_journal),
        "pipeline_agreement": rag_explanations.get("pipeline_agreement", True),
    }

    LOG_PATH = os.path.join(BASE_DIR, "data", "prediction_logs.jsonl")
    try:
        with open(LOG_PATH, "a") as f:
            f.write(str(log_entry) + "\n")
    except Exception:
        pass

    return {
        "spread": round(spread, 3),
        "journal_predictions": journal_predictions,
        "top3_recommendations": top3_recommendations,
        "rag_explanations": rag_explanations,
        "final_decision": final_decision,
        "recommended_headings": recommended_headings,
        "submission_recommendation": submission,
        "final_recommendation": final_recommendation,
        "best_journal": best_journal,
        "best_confidence": confidence,
    }
