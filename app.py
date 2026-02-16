"""
Flask web app for extractive Question Answering.
Input: passage (context) + question → output: answer span.
"""
import os
import sys

# Ensure project root and src are on path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2 MB max context

# Lazy-load model on first request
_qa_model = None


def get_qa_model():
    global _qa_model
    if _qa_model is None:
        from inference import QAInference
        model_path = os.path.join(ROOT, "outputs", "final")
        if os.path.isdir(model_path):
            _qa_model = QAInference(model_path=model_path)
        else:
            _qa_model = QAInference(model_name="distilbert-base-uncased")
    return _qa_model


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/answer", methods=["POST"])
def answer():
    data = request.get_json() or {}
    context = (data.get("context") or "").strip()
    question = (data.get("question") or "").strip()
    if not context or not question:
        return jsonify({"error": "Please enter both passage and question.", "answer": None}), 400
    try:
        model = get_qa_model()
        answer_text, score, start_char, end_char = model.predict(question, context)
        return jsonify({
            "answer": answer_text,
            "score": round(score, 4),
            "start_char": start_char,
            "end_char": end_char,
        })
    except Exception as e:
        return jsonify({"error": str(e), "answer": None}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
