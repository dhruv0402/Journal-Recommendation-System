import pickle
import numpy as np
import random

from src.phase2.learning_reranker import train

# -------- LOAD DATA --------
with open("reranker_training.pkl", "rb") as f:
    data = pickle.load(f)

if not data:
    raise ValueError(
        "Training data is empty. Run build_reranker_training_data.py first."
    )

training_data = []

# -------- PREPROCESS --------
for item in data:
    features = np.array(item["features"], dtype=np.float32)

    if features.ndim != 1 or len(features) == 0:
        continue

    training_data.append({"features": features, "label": int(item["label"])})

# shuffle
random.shuffle(training_data)

if len(training_data) == 0:
    raise ValueError("No valid training samples after preprocessing.")

print("Training samples:", len(training_data))

# -------- TRAIN --------
train(training_data, lr=0.01, epochs=300)

print("Model trained and saved")
