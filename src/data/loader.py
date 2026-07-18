# src/data/loader.py

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = BASE_DIR / "data" / "master_journals_expanded.csv"


def load_dataset() -> pd.DataFrame:
    """
    Loads the master journal dataset.

    Returns:
        pd.DataFrame with columns:
        - article_title
        - journal_name
        - abstract
    """
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH.resolve()}")

    df = pd.read_csv(DATASET_PATH)

    required_cols = {"article_title", "journal_name", "abstract"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    df = df.dropna(subset=["article_title", "journal_name", "abstract"])

    return df
