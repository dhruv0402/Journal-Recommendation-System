from pathlib import Path
import pandas as pd

from src.core.detector import detect_journal

DATASET_PATH = Path("data/master_journals_expanded.csv")


def run_pipeline(input_title: str):
    if not DATASET_PATH.exists():
        raise RuntimeError(f"Dataset not found at {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    return detect_journal(input_title, df)
