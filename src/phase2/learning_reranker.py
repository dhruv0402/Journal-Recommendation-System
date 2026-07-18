import numpy as np
import pickle
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "reranker_model.pkl")

_weights = None
_bias = 0.0


def load_model():
    global _weights, _bias

    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
            _weights = data.get("weights")
            _bias = data.get("bias", 0.0)
            print("[DEBUG] Reranker weights loaded:", _weights)
            print("[DEBUG] Reranker bias:", _bias)


def save_model(weights, bias):
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"weights": weights, "bias": bias}, f)


# ---------- CONSISTENT FEATURE FUNCTION ----------
def extract_features(j, rank=0):
    similarity = j.get("similarity", 0.0)
    confidence = j.get("confidence", similarity)

    return np.array(
        [
            similarity,
            confidence,
            similarity * confidence,
            (similarity**2),
            1.0 / (rank + 1.0),
            np.log(rank + 1.0),
        ],
        dtype=np.float32,
    )


# ---------- PREDICT ----------
def predict_score(j, rank=0):
    if _weights is None:
        return j.get("confidence", 0.0)

    x = extract_features(j, rank)
    return float(np.dot(_weights, x) + _bias)


# ---------- RERANK ----------
def rerank_with_learning(journal_predictions):
    if not journal_predictions:
        return journal_predictions

    if _weights is None:
        return journal_predictions

    for rank, j in enumerate(journal_predictions):
        j["confidence"] = predict_score(j, rank)

    journal_predictions.sort(key=lambda x: x["confidence"], reverse=True)

    return journal_predictions


# ---------- TRAIN ----------
def train(training_data, lr=0.01, epochs=100):
    global _weights, _bias

    X = []
    y = []

    for item in training_data:
        X.append(item["features"])
        y.append(item["label"])

    X = np.array(X)
    y = np.array(y)

    _weights = np.zeros(X.shape[1])
    _bias = 0.0

    for _ in range(epochs):
        preds = X @ _weights + _bias
        error = preds - y

        _weights -= lr * (X.T @ error) / len(X)
        _bias -= lr * np.mean(error)

    # FORCE model to care about similarity (bootstrap signal)
    if len(_weights) > 0:
        _weights[0] += 0.3
        _weights[2] += 0.2

    save_model(_weights, _bias)
