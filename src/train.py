"""
Fine-tune BERT/DistilBERT (or other) for SQuAD-style extractive question answering.
Evaluates with Exact Match and F1.
"""
import os
import argparse
from typing import Dict, List, Any

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from tqdm import tqdm

try:
    from .data_loader import load_squad_data
    from .preprocess import prepare_train_features, prepare_eval_features, squad_metrics
except ImportError:
    from data_loader import load_squad_data
    from preprocess import prepare_train_features, prepare_eval_features, squad_metrics


def get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_dataset(
    examples: List[Dict[str, Any]],
    tokenizer,
    max_length: int = 384,
    doc_stride: int = 128,
    batch_size: int = 8,
) -> Dataset:
    """Convert SQuAD examples to tokenized Dataset with start/end positions."""
    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []
    all_start_positions = []
    all_end_positions = []

    for i in tqdm(range(0, len(examples), batch_size), desc="Tokenizing"):
        batch = examples[i : i + batch_size]
        out = prepare_train_features(batch, tokenizer, max_length=max_length, doc_stride=doc_stride)
        n = len(out["input_ids"])
        all_input_ids.extend(out["input_ids"])
        all_attention_mask.extend(out["attention_mask"])
        if "token_type_ids" in out:
            all_token_type_ids.extend(out["token_type_ids"])
        all_start_positions.extend(out["start_positions"])
        all_end_positions.extend(out["end_positions"])

    d = {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "start_positions": all_start_positions,
        "end_positions": all_end_positions,
    }
    if all_token_type_ids:
        d["token_type_ids"] = all_token_type_ids
    return Dataset.from_dict(d)


def postprocess_qa_predictions(
    examples: List[Dict],
    features_with_mapping: List[Dict],
    raw_predictions: List[tuple],
    tokenizer,
    n_best_size: int = 20,
    max_answer_length: int = 30,
) -> Dict[str, str]:
    """
    Map predicted (start_logits, end_logits) back to answer text per example_id.
    raw_predictions: list of (start_logits, end_logits) for each feature.
    features_with_mapping: list of dicts with offset_mapping, example_id.
    """
    from collections import defaultdict

    example_id_to_index = {ex["id"]: i for i, ex in enumerate(examples)}
    id_to_answers = defaultdict(list)

    for feat_idx, (start_logits, end_logits) in enumerate(raw_predictions):
        if feat_idx >= len(features_with_mapping):
            break
        feat = features_with_mapping[feat_idx]
        example_id = feat["example_id"]
        offset_mapping = feat["offset_mapping"]
        if not offset_mapping:
            continue

        start_indexes = torch.tensor(start_logits).argsort(dim=-1, descending=True)[:n_best_size].tolist()
        end_indexes = torch.tensor(end_logits).argsort(dim=-1, descending=True)[:n_best_size].tolist()

        for start_idx in start_indexes:
            for end_idx in end_indexes:
                if start_idx >= len(offset_mapping) or end_idx >= len(offset_mapping):
                    continue
                if start_idx > end_idx:
                    continue
                if offset_mapping[start_idx] is None or offset_mapping[end_idx] is None:
                    continue
                if offset_mapping[start_idx][0] == 0 and offset_mapping[start_idx][1] == 0:
                    continue
                if offset_mapping[end_idx][0] == 0 and offset_mapping[end_idx][1] == 0:
                    continue
                length = end_idx - start_idx + 1
                if length > max_answer_length:
                    continue
                id_to_answers[example_id].append((start_idx, end_idx, start_logits[start_idx] + end_logits[end_idx]))

    # For each example_id take best (start, end) and decode
    predictions = {}
    for example_id, candidates in id_to_answers.items():
        if not candidates:
            predictions[example_id] = ""
            continue
        # Find the feature that contains this example_id to get offset_mapping and input_ids
        best_score = None
        best_start = best_end = None
        best_feat = None
        for feat in features_with_mapping:
            if feat["example_id"] != example_id:
                continue
            for (s, e, score) in candidates:
                if best_score is None or score > best_score:
                    best_score = score
                    best_start, best_end = s, e
                    best_feat = feat
            break
        if best_feat is None:
            predictions[example_id] = ""
            continue
        # Decode span from best_feat
        offset = best_feat["offset_mapping"][best_start]
        end_offset = best_feat["offset_mapping"][best_end]
        context_start = best_feat.get("context_start", 0)
        # Get context from original example
        ex_idx = example_id_to_index.get(example_id, 0)
        context = examples[ex_idx]["context"]
        if offset and end_offset:
            pred_text = context[offset[0] : end_offset[1]]
        else:
            pred_text = ""
        predictions[example_id] = pred_text

    return predictions


def run_evaluation(
    model,
    tokenizer,
    eval_examples: List[Dict],
    device,
    max_length: int = 384,
    doc_stride: int = 128,
    batch_size: int = 8,
) -> Dict[str, float]:
    """Run model on eval set and compute EM and F1."""
    model.eval()
    all_start_logits = []
    all_end_logits = []
    features_per_example = []  # list of (offset_mapping, example_id, context_start) per feature

    for i in tqdm(range(0, len(eval_examples), batch_size), desc="Eval"):
        batch = eval_examples[i : i + batch_size]
        tokenized, _ = prepare_eval_features(batch, tokenizer, max_length=max_length, doc_stride=doc_stride)
        # Keep offset_mapping for post-processing
        offset_mapping = tokenized.pop("offset_mapping", None)
        example_ids = tokenized.pop("example_id", None)

        inputs = {k: torch.tensor(v).to(device) for k, v in tokenized.items() if k != "overflow_to_sample_mapping"}
        with torch.no_grad():
            out = model(**inputs)
        start_logits = out.start_logits.cpu().tolist()
        end_logits = out.end_logits.cpu().tolist()

        for j in range(len(start_logits)):
            all_start_logits.append(start_logits[j])
            all_end_logits.append(end_logits[j])
            feat = {
                "example_id": example_ids[j] if example_ids else batch[0]["id"],
                "offset_mapping": offset_mapping[j] if offset_mapping else None,
            }
            features_per_example.append(feat)

    # Build predictions per example_id (handle multiple features per example)
    from collections import defaultdict
    example_id_to_ref = {ex["id"]: (ex["answers"][0]["text"] if ex.get("answers") else "") for ex in eval_examples}
    id_to_candidates = defaultdict(list)
    for idx, feat in enumerate(features_per_example):
        eid = feat["example_id"]
        start_logits = all_start_logits[idx]
        end_logits = all_end_logits[idx]
        offset_mapping = feat["offset_mapping"] or []
        start_idx = torch.tensor(start_logits).argmax().item()
        end_idx = torch.tensor(end_logits).argmax().item()
        if start_idx > end_idx or start_idx >= len(offset_mapping) or end_idx >= len(offset_mapping):
            pred_text = ""
        else:
            s, e = offset_mapping[start_idx], offset_mapping[end_idx]
            if s and e and (s[0] | e[0]) != 0:
                ex_idx = next(i for i, ex in enumerate(eval_examples) if ex["id"] == eid)
                ctx = eval_examples[ex_idx]["context"]
                pred_text = ctx[s[0] : e[1]]
            else:
                pred_text = ""
        id_to_candidates[eid].append((pred_text, start_logits[start_idx] + end_logits[end_idx]))

    predictions = {}
    for eid, candidates in id_to_candidates.items():
        best = max(candidates, key=lambda x: x[1])
        predictions[eid] = best[0]

    refs = [example_id_to_ref[eid] for eid in predictions]
    preds = [predictions[eid] for eid in predictions]
    return squad_metrics(preds, refs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--doc_stride", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=1000)
    args = parser.parse_args()

    root = get_project_root()
    data_dir = args.data_dir or os.path.join(root, "data")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data dir not found: {data_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_examples, dev_examples = load_squad_data(data_dir=data_dir)
    if args.max_train_samples:
        train_examples = train_examples[: args.max_train_samples]
    if args.max_eval_samples:
        dev_examples = dev_examples[: args.max_eval_samples]

    train_dataset = build_dataset(
        train_examples,
        tokenizer,
        max_length=args.max_length,
        doc_stride=args.doc_stride,
        batch_size=args.batch_size,
    )
    os.makedirs(args.output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
    )

    def compute_metrics(eval_pred):
        # Trainer passes (predictions, label_ids); for QA we need to run full eval with post-processing
        return {}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )
    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))

    # Evaluate with EM and F1
    metrics = run_evaluation(
        model, tokenizer, dev_examples, device,
        max_length=args.max_length, doc_stride=args.doc_stride, batch_size=args.batch_size,
    )
    print("Evaluation:", metrics)
    with open(os.path.join(args.output_dir, "eval_metrics.txt"), "w") as f:
        f.write(f"exact_match: {metrics['exact_match']}\nf1: {metrics['f1']}\n")
    return metrics


if __name__ == "__main__":
    main()
