"""
Load fine-tuned QA model and extract answer span from context + question.
"""
import os
from typing import Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


def get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class QAInference:
    """Question answering inference using a fine-tuned model."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = "distilbert-base-uncased",
        device: Optional[str] = None,
    ):
        """
        Load model and tokenizer.
        model_path: path to fine-tuned model dir (with config.json, pytorch_model.bin). If None, use model_name.
        model_name: HuggingFace model name if model_path is None.
        """
        if model_path and os.path.isdir(model_path):
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, question: str, context: str) -> Tuple[str, float, int, int]:
        """
        Extract answer span from context given the question.
        Returns (answer_text, score, start_char, end_char).
        """
        inputs = self.tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=512,
            padding=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offset_mapping = inputs.pop("offset_mapping", None)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        start_logits = outputs.start_logits[0].cpu()
        end_logits = outputs.end_logits[0].cpu()
        start_idx = start_logits.argmax().item()
        end_idx = end_logits.argmax().item()
        score = (start_logits[start_idx] + end_logits[end_idx]).item()

        if offset_mapping is None:
            return "", float(score), 0, 0
        offsets = offset_mapping[0].cpu().tolist()
        if start_idx >= len(offsets) or end_idx >= len(offsets) or start_idx > end_idx:
            return "", float(score), 0, 0
        start_char = offsets[start_idx][0]
        end_char = offsets[end_idx][1]
        if start_char == 0 and end_char == 0:
            return "", float(score), 0, 0
        answer_text = context[start_char:end_char]
        return answer_text.strip(), float(score), start_char, end_char


def load_qa_model(model_path: Optional[str] = None) -> QAInference:
    """Convenience: load from project outputs/final if exists, else use default."""
    root = get_project_root()
    default_path = os.path.join(root, "outputs", "final")
    path = model_path or (default_path if os.path.isdir(default_path) else None)
    return QAInference(model_path=path, model_name="distilbert-base-uncased")
