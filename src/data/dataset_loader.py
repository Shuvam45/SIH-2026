from pathlib import Path
import pandas as pd

DATASET_PATH = Path("/app/data/WEB_READY_ULPIN_MASTER_DATASET.csv")


def get_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    return pd.read_csv(DATASET_PATH)