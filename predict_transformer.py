#!/usr/bin/env python3
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def predict(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()

    inputs = tokenizer(args.text, return_tensors="pt", truncation=True, max_length=args.max_length)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = logits.argmax(dim=-1).item()
    label = model.config.id2label[str(pred)] if isinstance(model.config.id2label, dict) else model.config.id2label[pred]
    print(label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Uzbek news category (transformer)")
    parser.add_argument("--model-dir", default="artifacts_transformer")
    parser.add_argument("--text", required=True)
    parser.add_argument("--max-length", type=int, default=256)
    predict(parser.parse_args())
