import pandas as pd
import numpy as np
import pickle
import random


from src.phase2.optimized_phase2 import run_phase2_fast, preload_phase2

DATA_PATH = "data/master_journals_expanded.csv"

df = pd.read_csv(DATA_PATH)

#  Initialize FAISS before training loop
preload_phase2(df)

training_data = []

for i, row in df.iterrows():
    abstract = str(row["abstract"])
    true_journal = row["journal_name"]

    try:
        result = run_phase2_fast(abstract, training_mode=True)
        preds = result.get("journal_predictions", [])

        for rank, j in enumerate(preds[:20]):
            similarity = j.get("similarity", 0.0)
            confidence = j.get("confidence", similarity)

            features = [
                similarity,
                confidence,
                similarity * confidence,
                (similarity**2),
                1.0 / (rank + 1.0),
                np.log(rank + 1.0),
            ]

            if j["journal_name"] == true_journal:
                label = 1.0
            elif rank < 3:
                label = 0.3
            elif rank < 10:
                label = 0.1
            else:
                label = 0.0

            # keep all samples (no filtering — full ranking signal)
            training_data.append({"features": features, "label": label})

    except Exception as e:
        print(f"[ERROR] Row {i} failed:", e)
        continue

    if i % 200 == 0:
        print(f"Processed {i}")

# shuffle
random.shuffle(training_data)

# save
with open("reranker_training.pkl", "wb") as f:
    pickle.dump(training_data, f)

print("Training data saved:", len(training_data))
