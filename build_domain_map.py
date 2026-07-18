import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

# Load dataset
df = pd.read_csv("data/master_journals_expanded.csv")

model = SentenceTransformer("all-MiniLM-L6-v2")

# Define domain anchors
DOMAIN_TEXTS = {
    "systems": "distributed systems parallel computing scheduling architecture",
    "networks": "computer networks routing communication protocols wireless",
    "software": "software engineering testing development system design",
    "AI": "machine learning deep learning neural networks artificial intelligence",
    "data": "data science databases big data analytics data mining",
    "interdisciplinary": "cross domain multidisciplinary applied research",
}

DOMAIN_EMBS = {
    k: model.encode(v, normalize_embeddings=True) for k, v in DOMAIN_TEXTS.items()
}


def detect_domain(text):
    emb = model.encode(text, normalize_embeddings=True)

    best_domain = None
    best_score = -1

    for domain, d_emb in DOMAIN_EMBS.items():
        score = float(np.dot(emb, d_emb))
        if score > best_score:
            best_score = score
            best_domain = domain

    return best_domain


journal_domain_map = {}

for _, row in df.iterrows():
    journal = row["journal_name"]
    text = str(row.get("abstract", ""))  # or scope if exists

    domain = detect_domain(text)
    journal_domain_map[journal] = domain


# Save to file
with open("journal_domain_map.py", "w") as f:
    f.write("journal_domain_map = {\n")
    for k, v in journal_domain_map.items():
        f.write(f'    "{k}": "{v}",\n')
    f.write("}\n")

print("Domain map generated → journal_domain_map.py")
