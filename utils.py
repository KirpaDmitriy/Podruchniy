import json
from pathlib import Path

import pandas as pd


DATA_DIR = "./data"
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)


def get_data_path(file_id: str) -> str:
    return f"{DATA_DIR}/{file_id}.csv"


def get_hypothesis_path(hypo_id: str) -> str:
    return f"{DATA_DIR}/{hypo_id}-HYPO.json"


def load_hypothesis(file_id: str) -> dict:
    with open(get_hypothesis_path(file_id), "r") as f:
        return json.load(f)


def save_data(file_id: str, data: bytes) -> None:
    with open(get_data_path(file_id), "wb") as f:
        f.write(data)


def save_hypothesis(hypo_id: str, hypothesis: dict) -> None:
    with open(get_hypothesis_path(hypo_id), "w") as f:
        json.dump(hypothesis, f)


def load_data_file(data_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(data_path, encoding="utf8")
    except Exception as load_error:
        print(f"Error while loading from raw: {load_error}")
        return pd.read_csv(data_path, encoding="latin")