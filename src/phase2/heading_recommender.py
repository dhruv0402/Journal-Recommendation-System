from collections import Counter
from typing import List, Dict
import re


def tokenize(text: str) -> list:
    """Simple whitespace + punctuation tokenizer (replaces deleted bm25_extractor)."""
    return re.findall(r"\b[a-z]{2,}\b", text.lower())


def recommend_heading(
    bm25,
    corpus_tokens,
    df,
    user_abstract: str,
    top_k: int = 20
) -> Dict:
    """
    Recommends:
    - Best journal name
    - Dominant domain keywords inside that journal
    """

    query_tokens = tokenize(user_abstract)
    scores = bm25.get_scores(query_tokens)

    # Top-K abstract indices
    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    journal_counter = Counter()
    domain_tokens = []

    for idx in top_indices:
        row = df.iloc[idx]
        journal = str(row.get("journal_name", "")).strip()
        abstract = str(row.get("abstract", ""))

        if journal:
            journal_counter[journal] += 1
            domain_tokens.extend(tokenize(abstract))

    if not journal_counter:
        return {}

    # Primary journal
    primary_journal = journal_counter.most_common(1)[0][0]

    # Domain keywords (inside that journal)
    domain_counter = Counter(domain_tokens)
    domain_keywords = [
        k for k, _ in domain_counter.most_common(5)
    ]

    return {
        "journal_name": primary_journal,
        "suggested_domain": " / ".join(domain_keywords),
        "confidence": round(journal_counter[primary_journal] / top_k, 2)
    }