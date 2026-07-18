"""
Full evaluation with baseline comparisons.

Runs 4 conditions on the same held-out test set:
  1. BM25 only          — pure keyword search, no embeddings
  2. FAISS only         — embeddings + retrieval, no reranker
  3. Full pipeline      — FAISS + scope reranker + learning reranker
  4. Full + validator   — Full pipeline + LLM-as-validator override

Produces a comparison table for the paper.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from rank_bm25 import BM25Okapi

# ── import pipeline components directly ──────────────────────────────────────
from src.phase2.optimized_phase2 import preload_phase2, run_phase2_fast, _model
from src.phase2.abstract_aggregation import aggregate_abstract_results
from src.phase2.scope_reranker import rerank_with_scope
from src.phase2.learning_reranker import rerank_with_learning

# ── CONFIG ───────────────────────────────────────────────────────────────────
DATA_PATH = "data/master_journals_expanded.csv"
HELD_OUT_N = 500  # target test size (stratified)
RANDOM_SEED = 99  # different from training seed (42)
TOP_K = 3

# ── LOAD DATA ────────────────────────────────────────────────────────────────
df = (
    pd.read_csv(DATA_PATH)
    .dropna(subset=["abstract", "journal_name"])
    .reset_index(drop=True)
)
print(f"Total rows: {len(df)}")

# ── STRATIFIED HELD-OUT SPLIT ────────────────────────────────────────────────
test_rows = []
for journal in df["journal_name"].unique():
    jdf = df[df["journal_name"] == journal]
    n = max(2, min(50, int(len(jdf) * HELD_OUT_N / len(df))))
    test_rows.append(jdf.sample(n=min(n, len(jdf)), random_state=RANDOM_SEED))

test_df = pd.concat(test_rows)
train_df = df.drop(test_df.index).reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(f"Train: {len(train_df)}  |  Test: {len(test_df)}")
print(f"Journals in test: {test_df['journal_name'].nunique()}")
print(f"\nJournal distribution:\n{test_df['journal_name'].value_counts()}\n")

# ── PRELOAD FULL PIPELINE (train only — no leakage) ──────────────────────────
print("Preloading pipeline on train set...")
preload_phase2(train_df, force_rebuild=True)

# ── BUILD BM25 INDEX ─────────────────────────────────────────────────────────
print("Building BM25 index...")
train_abstracts = train_df["abstract"].fillna("").tolist()
train_journals = train_df["journal_name"].tolist()
tokenized = [a.lower().split() for a in train_abstracts]
bm25 = BM25Okapi(tokenized)
print("BM25 ready.\n")


# ── METRICS ──────────────────────────────────────────────────────────────────
def hit_at_k(preds, truth, k):
    return int(truth in [p["journal_name"] for p in preds[:k]])


def reciprocal_rank(preds, truth):
    for i, p in enumerate(preds):
        if p["journal_name"] == truth:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(preds, truth, k):
    for i, p in enumerate(preds[:k]):
        if p["journal_name"] == truth:
            return 1.0 / np.log2(i + 2)
    return 0.0


def mean(arr):
    return round(sum(arr) / len(arr), 4) if arr else 0.0


def score(preds, truth):
    return {
        "h1": hit_at_k(preds, truth, 1),
        "h3": hit_at_k(preds, truth, 3),
        "mrr": reciprocal_rank(preds, truth),
        "ndcg": ndcg_at_k(preds, truth, 3),
    }


# ── CONDITION 1: BM25 ONLY ───────────────────────────────────────────────────
def run_bm25(abstract):
    tokens = abstract.lower().split()
    scores = bm25.get_scores(tokens)
    top_idx = np.argsort(scores)[::-1][:50]

    from collections import defaultdict

    journal_scores = defaultdict(list)
    for idx in top_idx:
        journal_scores[train_journals[idx]].append(float(scores[idx]))

    agg = []
    for journal, sc in journal_scores.items():
        agg.append(
            {"journal_name": journal, "confidence": max(sc), "similarity": max(sc)}
        )

    return sorted(agg, key=lambda x: x["confidence"], reverse=True)


# ── CONDITION 2: FAISS ONLY (no rerankers) ───────────────────────────────────
def run_faiss_only(abstract):
    from src.phase2.optimized_phase2 import _index as faiss_idx, _metadata as faiss_meta
    emb = _model.encode([abstract], normalize_embeddings=True)
    scores, indices = faiss_idx.search(emb, 50)

    article_results = []
    for sc, idx in zip(scores[0], indices[0]):
        _, journal = faiss_meta[idx]
        article_results.append(
            {
                "journal_name": journal,
                "similarity": round(max(0.0, min(1.0, float(sc))), 3),
            }
        )

    return aggregate_abstract_results(article_results)


# ── CONDITION 3: FULL PIPELINE (FAISS + rerankers, no validator) ─────────────
def run_full_pipeline(abstract):
    out = run_phase2_fast(abstract, top_k=50)
    preds = out.get("journal_predictions", [])
    return sorted(preds, key=lambda x: x.get("confidence", 0.0), reverse=True)


# ── CONDITION 4: FULL + VALIDATOR ────────────────────────────────────────────
# We track when the LLM validator overrides the pipeline and whether
# that override improves or hurts Hit@1.
from src.rag.rag_engine import RAGEngine as _RAGEngine

_rag = None


def _get_rag():
    global _rag
    if _rag is None:
        _rag = _RAGEngine(train_df)
    return _rag


def run_with_validator(abstract, pipeline_preds):
    """
    Takes the top-3 pipeline predictions, asks the LLM validator to
    independently pick the best journal, and returns adjusted predictions
    with the LLM pick promoted to rank 1 if it disagrees.
    """
    top3 = pipeline_preds[:3]
    try:
        rag = _get_rag()
        result = rag.generate(abstract, top3)
        llm_pick = result.get("best_journal", "")
        pipeline_top = top3[0].get("journal_name", "") if top3 else ""
        agreed = result.get("pipeline_agreement", True)

        if not agreed and llm_pick:
            # Promote LLM pick to rank 1 if it's in the candidate list
            reranked = []
            promoted = None
            for p in pipeline_preds:
                if p["journal_name"].lower() == llm_pick.lower():
                    promoted = p
                else:
                    reranked.append(p)
            if promoted:
                return [promoted] + reranked, True  # True = override happened
        return pipeline_preds, False
    except Exception as e:
        print(f"    [validator error] {e}")
        return pipeline_preds, False


# ── MAIN EVALUATION LOOP ─────────────────────────────────────────────────────
conditions = {
    "BM25": {"h1": [], "h3": [], "mrr": [], "ndcg": []},
    "FAISS_only": {"h1": [], "h3": [], "mrr": [], "ndcg": []},
    "Full_pipeline": {"h1": [], "h3": [], "mrr": [], "ndcg": []},
    "Full+Validator": {"h1": [], "h3": [], "mrr": [], "ndcg": []},
}

per_journal = {c: {} for c in conditions}
validator_overrides = {"total": 0, "improved": 0, "hurt": 0, "neutral": 0}
skipped = 0

print("=" * 60)
print("Running evaluation across 4 conditions...")
print("=" * 60)

for i, (_, row) in enumerate(test_df.iterrows()):
    abstract = row["abstract"]
    truth = row["journal_name"]

    try:
        # ── condition 1: BM25 ──
        bm25_preds = run_bm25(abstract)
        s = score(bm25_preds, truth)
        for k, v in s.items():
            conditions["BM25"][k].append(v)

        # ── condition 2: FAISS only ──
        faiss_preds = run_faiss_only(abstract)
        s = score(faiss_preds, truth)
        for k, v in s.items():
            conditions["FAISS_only"][k].append(v)

        # ── condition 3: full pipeline ──
        full_preds = run_full_pipeline(abstract)
        s = score(full_preds, truth)
        for k, v in s.items():
            conditions["Full_pipeline"][k].append(v)
        full_h1 = s["h1"]

        # ── condition 4: full + validator ──
        val_preds, overridden = run_with_validator(abstract, full_preds)
        s = score(val_preds, truth)
        for k, v in s.items():
            conditions["Full+Validator"][k].append(v)
        val_h1 = s["h1"]

        # Track validator impact
        if overridden:
            validator_overrides["total"] += 1
            if val_h1 > full_h1:
                validator_overrides["improved"] += 1
            elif val_h1 < full_h1:
                validator_overrides["hurt"] += 1
            else:
                validator_overrides["neutral"] += 1

        # Per-journal tracking (full pipeline)
        if truth not in per_journal["Full_pipeline"]:
            for c in conditions:
                per_journal[c][truth] = {"h1": [], "h3": [], "mrr": []}

        for c, preds in [
            ("BM25", bm25_preds),
            ("FAISS_only", faiss_preds),
            ("Full_pipeline", full_preds),
            ("Full+Validator", val_preds),
        ]:
            s = score(preds, truth)
            per_journal[c][truth]["h1"].append(s["h1"])
            per_journal[c][truth]["h3"].append(s["h3"])
            per_journal[c][truth]["mrr"].append(s["mrr"])

    except Exception as e:
        print(f"  [SKIP row {i}]: {e}")
        skipped += 1
        continue

    if (i + 1) % 50 == 0:
        fp = conditions["Full_pipeline"]
        print(
            f"  [{i + 1}/{len(test_df)}] Full pipeline → "
            f"Hit@1={mean(fp['h1']):.3f}  Hit@3={mean(fp['h3']):.3f}  MRR={mean(fp['mrr']):.3f}"
        )

# ── RESULTS TABLE ────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("                    COMPARISON RESULTS")
print("=" * 72)
print(f"{'Condition':<22} {'Hit@1':>7} {'Hit@3':>7} {'MRR':>7} {'nDCG@3':>8}")
print("-" * 72)

for cname, scores in conditions.items():
    n = len(scores["h1"])
    print(
        f"  {cname:<20} {mean(scores['h1']) * 100:>6.1f}% {mean(scores['h3']) * 100:>6.1f}%"
        f" {mean(scores['mrr']):>7.4f} {mean(scores['ndcg']):>8.4f}   (n={n})"
    )

print("=" * 72)

# ── VALIDATOR ANALYSIS ───────────────────────────────────────────────────────
total_tested = len(test_df) - skipped
print(f"\nValidator Override Analysis (out of {total_tested} samples):")
print(f"  Total overrides    : {validator_overrides['total']}")
print(f"  Improved Hit@1     : {validator_overrides['improved']}")
print(f"  Hurt Hit@1         : {validator_overrides['hurt']}")
print(f"  No change          : {validator_overrides['neutral']}")
if validator_overrides["total"] > 0:
    precision = validator_overrides["improved"] / validator_overrides["total"]
    print(
        f"  Override precision : {precision:.1%}  (% of overrides that improved result)"
    )

# ── PER-JOURNAL TABLE (full pipeline) ────────────────────────────────────────
print(f"\nPer-Journal Results (Full Pipeline):")
print(f"  {'Journal':<45} {'n':>4} {'Hit@1':>6} {'Hit@3':>6} {'MRR':>6}")
print("  " + "-" * 68)

fp_pj = per_journal["Full_pipeline"]
for journal, s in sorted(fp_pj.items(), key=lambda x: -mean(x[1]["h1"])):
    n = len(s["h1"])
    print(
        f"  {journal[:44]:<44} {n:>4}  {mean(s['h1']) * 100:>5.1f}%"
        f"  {mean(s['h3']) * 100:>5.1f}%  {mean(s['mrr']):>5.3f}"
    )

# ── SAVE ─────────────────────────────────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = f"outputs/eval_full_{timestamp}.json"

summary = {
    "timestamp": timestamp,
    "dataset_size": len(df),
    "train_size": len(train_df),
    "test_size": len(test_df),
    "skipped": skipped,
    "conditions": {
        c: {
            "hit_at_1": mean(s["h1"]),
            "hit_at_3": mean(s["h3"]),
            "mrr": mean(s["mrr"]),
            "ndcg_at_3": mean(s["ndcg"]),
            "n": len(s["h1"]),
        }
        for c, s in conditions.items()
    },
    "validator_overrides": validator_overrides,
    "per_journal_full_pipeline": {
        j: {
            "n": len(s["h1"]),
            "hit@1": mean(s["h1"]),
            "hit@3": mean(s["h3"]),
            "mrr": mean(s["mrr"]),
        }
        for j, s in fp_pj.items()
    },
}

with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSaved → {out_path}")
