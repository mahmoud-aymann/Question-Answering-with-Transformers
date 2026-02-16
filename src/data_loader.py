"""
Load and preprocess Stanford Question Answering Dataset (SQuAD) v1.1
"""
import json
import os
from typing import List, Dict, Any, Optional

import pandas as pd


def load_squad_from_json(path: str) -> List[Dict[str, Any]]:
    """
    Load SQuAD format JSON and flatten to list of examples.
    Each example: {id, title, context, question, answers: [{text, answer_start}]}
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = []
    for article in data["data"]:
        title = article.get("title", "")
        for para in article["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                examples.append({
                    "id": qa["id"],
                    "title": title,
                    "context": context,
                    "question": qa["question"],
                    "answers": qa["answers"],
                })
    return examples


def squad_to_dataframe(examples: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert list of SQuAD examples to DataFrame (one answer per row; use first answer for span)."""
    rows = []
    for ex in examples:
        ans = ex["answers"][0] if ex["answers"] else {"text": "", "answer_start": 0}
        rows.append({
            "id": ex["id"],
            "title": ex["title"],
            "context": ex["context"],
            "question": ex["question"],
            "answer_text": ans["text"],
            "answer_start": ans["answer_start"],
        })
    return pd.DataFrame(rows)


def load_squad_data(
    data_dir: str = "data",
    train_file: str = "train-v1.1.json",
    dev_file: str = "dev-v1.1.json",
) -> tuple:
    """
    Load train and dev SQuAD data from directory.
    Returns (train_examples, dev_examples) as lists of dicts.
    """
    train_path = os.path.join(data_dir, train_file)
    dev_path = os.path.join(data_dir, dev_file)
    if not os.path.isfile(train_path):
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not os.path.isfile(dev_path):
        raise FileNotFoundError(f"Dev file not found: {dev_path}")

    train_examples = load_squad_from_json(train_path)
    dev_examples = load_squad_from_json(dev_path)
    return train_examples, dev_examples


if __name__ == "__main__":
    import sys
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base, "data")
    train, dev = load_squad_data(data_dir=data_dir)
    print(f"Train examples: {len(train)}, Dev examples: {len(dev)}")
    if train:
        print("Sample:", json.dumps(train[0], indent=2)[:500])
