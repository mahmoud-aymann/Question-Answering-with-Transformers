# Question Answering with Transformers

Extractive question answering system using Transformer models fine-tuned on the Stanford Question Answering Dataset (SQuAD), with a Flask web interface.

## Requirements

- Python 3.8+
- SQuAD data: place `train-v1.1.json` and `dev-v1.1.json` inside a folder named `data/` (e.g. download from [Kaggle](https://www.kaggle.com/datasets/stanfordedu/squad)).

## Installation

```bash
pip install -r requirements.txt
```

## Training (fine-tuning)

Training on the full dataset can take a while. Start with a smaller subset for quick runs:

```bash
# Quick run (small subset)
python -m src.train --model_name distilbert-base-uncased --max_train_samples 2000 --max_eval_samples 500 --epochs 2 --output_dir ./outputs

# Full training
python -m src.train --model_name distilbert-base-uncased --epochs 3 --output_dir ./outputs
```

Arguments: `--model_name`, `--data_dir`, `--output_dir`, `--epochs`, `--batch_size`, etc.  
Metrics are saved to `outputs/eval_metrics.txt` (Exact Match and F1).

## Running the Flask app

```bash
python app.py
```

Open **http://127.0.0.1:5000**. The app uses `outputs/final` if present, otherwise the base model.

## Project structure

```
├── app.py
├── requirements.txt
├── README.md
├── question_answering.ipynb
├── data/                      # add train-v1.1.json, dev-v1.1.json
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── train.py
│   └── inference.py
├── templates/
│   ├── base.html
│   └── index.html
├── static/
│   └── css/
│       └── style.css
└── outputs/                  # created after training
    └── final/
```

## Tools

- Hugging Face Transformers, Tokenizers
- Pandas, Datasets
- Flask
- Evaluation: Exact Match and F1

## Bonus: different base models

```bash
python -m src.train --model_name bert-base-uncased --output_dir ./outputs_bert
python -m src.train --model_name distilbert-base-uncased --output_dir ./outputs_distilbert
```

Compare `eval_metrics.txt` in each output folder.
