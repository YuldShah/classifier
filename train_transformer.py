#!/usr/bin/env python3
import argparse
import os
from typing import Optional

from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def _pick_text_columns(columns):
    primary = "text" if "text" in columns else columns[0]
    secondary = "title" if "title" in columns else None
    return primary, secondary


def _pick_label_column(columns):
    for candidate in ("label", "category", "folder", "class"):
        if candidate in columns:
            return candidate
    raise ValueError(
        f"Could not infer label column from {columns}. Set --label-col explicitly."
    )


def _join_texts(example, text_col, title_col):
    text = example[text_col] or ""
    title = example[title_col] if title_col else ""
    combined = f"{title} {text}".strip() if title else text.strip()
    return combined


def main(args: argparse.Namespace) -> None:
    ds = load_dataset("MLDataScientist/Uzbek_news_dataset")
    split = args.split or ("train" if "train" in ds else list(ds.keys())[0])
    data = ds[split]

    columns = list(data.column_names)
    text_col = args.text_col or _pick_text_columns(columns)[0]
    title_col = args.title_col or _pick_text_columns(columns)[1]
    label_col = args.label_col or _pick_label_column(columns)

    if args.shuffle:
        data = data.shuffle(seed=args.seed)

    if args.max_samples is not None:
        start = args.start or 0
        end = start + args.max_samples
        data = data.select(range(start, end))

    if args.resume_from_checkpoint:
        config = AutoConfig.from_pretrained(args.resume_from_checkpoint)
        id2label = {
            int(k) if isinstance(k, str) and k.isdigit() else int(k): v
            for k, v in config.id2label.items()
        }
        label2id = {v: k for k, v in id2label.items()}
    else:
        label_source = args.label_source
        if label_source == "full":
            all_labels = ds[split][label_col]
            labels = sorted(set(all_labels))
        else:
            labels = sorted(set(data[label_col]))
        label2id = {label: idx for idx, label in enumerate(labels)}
        id2label = {idx: label for label, idx in label2id.items()}

    def preprocess(example):
        text = _join_texts(example, text_col, title_col)
        label_value = example[label_col]
        if label_value not in label2id:
            raise ValueError(
                f"Label '{label_value}' not in label set. "
                "Use --label-source full or ensure label mapping matches checkpoint."
            )
        return {
            "text": text,
            "label": label2id[label_value],
        }

    data = data.map(preprocess, remove_columns=data.column_names)

    train_test = data.train_test_split(test_size=args.val_size, seed=42)
    train_ds = train_test["train"]
    val_ds = train_test["test"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
        )

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        problem_type="single_label_classification",
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="weighted"),
        }

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=args.fp16,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train transformer Uzbek news classifier")
    parser.add_argument("--model-name", default="xlm-roberta-base")
    parser.add_argument("--text-col", default=None)
    parser.add_argument("--title-col", default=None)
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--out-dir", default="artifacts_transformer")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument(
        "--label-source",
        choices=["subset", "full"],
        default="subset",
        help="Build label set from subset or full dataset",
    )
    main(parser.parse_args())
