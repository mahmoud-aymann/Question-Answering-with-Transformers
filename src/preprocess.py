"""
Preprocess SQuAD for Hugging Face Question Answering (tokenization + span alignment).
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from transformers import AutoTokenizer


@dataclass
class SquadExample:
    """Single SQuAD example with answer span in character offsets."""
    id: str
    context: str
    question: str
    answer_text: str
    answer_start: int

    @property
    def answer_end(self) -> int:
        return self.answer_start + len(self.answer_text)


def prepare_train_features(
    examples: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_length: int = 384,
    doc_stride: int = 128,
) -> Dict[str, List]:
    """
    Tokenize and align answer spans. Handles long contexts by sliding window (doc_stride).
    Returns dict of lists: input_ids, attention_mask, token_type_ids, start_positions, end_positions, etc.
    """
    questions = [ex["question"] for ex in examples]
    contexts = [ex["context"] for ex in examples]
    answer_starts = []
    answer_texts = []
    for ex in examples:
        ans = ex["answers"][0] if ex.get("answers") else {"text": "", "answer_start": 0}
        answer_starts.append(ans["answer_start"])
        answer_texts.append(ans["text"])

    tokenized = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors=None,
    )

    # Map character answer span to token span
    sample_mapping = tokenized.pop("overflow_to_sample_mapping", None)
    offset_mapping = tokenized.pop("offset_mapping", None)

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_mapping[i] if sample_mapping is not None else 0
        start_char = answer_starts[sample_idx]
        end_char = start_char + len(answer_texts[sample_idx])

        seq_ids = tokenized["token_type_ids"][i] if "token_type_ids" in tokenized else [0] * len(offsets)

        start_token = 0
        end_token = 0
        for idx, (s, e) in enumerate(offsets):
            if s == 0 and e == 0:
                continue
            if seq_ids[idx] != 1:  # context tokens usually have type 1
                continue
            if start_char >= e or end_char <= s:
                continue
            if start_char >= s and start_char < e:
                start_token = idx
            if end_char > s and end_char <= e:
                end_token = idx
                break
            if end_char > e:
                end_token = idx

        start_positions.append(start_token)
        end_positions.append(end_token)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized


def prepare_eval_features(
    examples: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_length: int = 384,
    doc_stride: int = 128,
) -> Tuple[Dict, List[Dict]]:
    """
    Tokenize for evaluation (no labels). Keep offset_mapping and example_id for later decoding.
    Returns (tokenized_batch, list of {example_id, context, question, answers}).
    """
    tokenized = tokenizer(
        [ex["question"] for ex in examples],
        [ex["context"] for ex in examples],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors=None,
    )
    sample_mapping = tokenized.get("overflow_to_sample_mapping", [0] * len(tokenized["input_ids"]))
    # Store for post-processing
    tokenized["example_id"] = []
    for i in range(len(tokenized["input_ids"])):
        tokenized["example_id"].append(examples[sample_mapping[i]]["id"])
    return tokenized, examples


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    import string
    import re
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text, flags=re.IGNORECASE)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return ''.join(c for c in text if c not in string.punctuation)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """1.0 if normalized prediction equals normalized ground truth else 0."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1: overlap between prediction and ground truth tokens."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = sum(1 for t in pred_tokens if t in gold_tokens)
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def squad_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute exact match and F1 over list of (prediction, reference) pairs."""
    em = np.mean([exact_match_score(p, r) for p, r in zip(predictions, references)])
    f1 = np.mean([f1_score(p, r) for p, r in zip(predictions, references)])
    return {"exact_match": float(em), "f1": float(f1)}
