#!/usr/bin/env python3
import argparse
import itertools
import os
from typing import Iterable, List, Optional, Tuple

from datasets import load_dataset
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import joblib


def _pick_text_columns(columns: List[str]) -> Tuple[str, Optional[str]]:
    primary = "text" if "text" in columns else columns[0]
    secondary = "title" if "title" in columns else None
    return primary, secondary


def _pick_label_column(columns: List[str]) -> str:
    for candidate in ("label", "category", "folder", "class"):
        if candidate in columns:
            return candidate
    raise ValueError(
        f"Could not infer label column from {columns}. Set --label-col explicitly."
    )


def _build_texts(primary: List[str], secondary: Optional[List[str]]) -> List[str]:
    if secondary is None:
        return primary
    return [f"{t} {s}".strip() for t, s in zip(primary, secondary)]


def _iter_text_label(
    dataset: Iterable[dict], text_col: str, title_col: Optional[str], label_col: str
) -> Iterable[Tuple[str, str]]:
    for item in dataset:
        text = item[text_col] or ""
        title = item[title_col] if title_col else ""
        combined = f"{title} {text}".strip() if title else text.strip()
        yield combined, item[label_col]


def _batch_iter(
    iterator: Iterable[Tuple[str, str]], batch_size: int
) -> Iterable[Tuple[List[str], List[str]]]:
    batch_texts: List[str] = []
    batch_labels: List[str] = []
    for text, label in iterator:
        batch_texts.append(text)
        batch_labels.append(label)
        if len(batch_texts) >= batch_size:
            yield batch_texts, batch_labels
            batch_texts, batch_labels = [], []
    if batch_texts:
        yield batch_texts, batch_labels


def train(args: argparse.Namespace) -> None:
    ds = load_dataset("MLDataScientist/Uzbek_news_dataset", streaming=args.streaming)
    split = args.split or ("train" if "train" in ds else list(ds.keys())[0])
    data = ds[split]

    columns = list(data.column_names)
    text_col = args.text_col or _pick_text_columns(columns)[0]
    title_col = args.title_col or _pick_text_columns(columns)[1]
    label_col = args.label_col or _pick_label_column(columns)

    label_encoder = LabelEncoder()

    if args.streaming:
        if args.max_train is None or args.max_val is None:
            raise ValueError("--max-train and --max-val are required for --streaming")

        total_needed = args.max_train + args.max_val
        label_sample = [
            item[label_col]
            for item in itertools.islice(ds[split], total_needed)
        ]
        classes = sorted(set(label_sample))
        label_encoder.fit(classes)

        stream_iter = _iter_text_label(ds[split], text_col, title_col, label_col)
        val_examples = list(itertools.islice(stream_iter, args.max_val))
        train_iter = itertools.islice(stream_iter, args.max_train)

        vectorizer = HashingVectorizer(
            n_features=args.hash_features,
            ngram_range=(1, args.max_ngram),
            alternate_sign=False,
            norm="l2",
        )
        clf = SGDClassifier(
            loss="log_loss",
            alpha=args.alpha,
            max_iter=1,
            tol=None,
            random_state=42,
        )

        class_ids = list(range(len(label_encoder.classes_)))
        for idx, (texts, labels) in enumerate(
            _batch_iter(train_iter, args.batch_size)
        ):
            X = vectorizer.transform(texts)
            y = label_encoder.transform(labels)
            if idx == 0:
                clf.partial_fit(X, y, classes=class_ids)
            else:
                clf.partial_fit(X, y)

        val_texts, val_labels = zip(*val_examples) if val_examples else ([], [])
        X_val = vectorizer.transform(list(val_texts))
        y_val = label_encoder.transform(list(val_labels)) if val_labels else []
        preds = clf.predict(X_val) if len(val_texts) else []
        acc = accuracy_score(y_val, preds) if len(val_texts) else 0.0
        f1 = f1_score(y_val, preds, average="weighted") if len(val_texts) else 0.0

        os.makedirs(args.out_dir, exist_ok=True)
        joblib.dump(
            {"vectorizer": vectorizer, "classifier": clf},
            os.path.join(args.out_dir, "model.joblib"),
        )
        joblib.dump(label_encoder, os.path.join(args.out_dir, "label_encoder.joblib"))
    else:
        texts = _build_texts(data[text_col], data[title_col] if title_col else None)
        labels = data[label_col]
        y = label_encoder.fit_transform(labels)

        X_train, X_val, y_train, y_val = train_test_split(
            texts, y, test_size=args.val_size, random_state=42, stratify=y
        )

        model = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=args.max_features,
                        ngram_range=(1, args.max_ngram),
                        min_df=args.min_df,
                    ),
                ),
                (
                    "clf",
                    SGDClassifier(
                        loss="log_loss",
                        alpha=args.alpha,
                        max_iter=1000,
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds, average="weighted")

        os.makedirs(args.out_dir, exist_ok=True)
        joblib.dump(model, os.path.join(args.out_dir, "model.joblib"))
        joblib.dump(label_encoder, os.path.join(args.out_dir, "label_encoder.joblib"))

    print(f"Validation accuracy: {acc:.4f}")
    print(f"Validation weighted F1: {f1:.4f}")
    print(f"Saved model to {args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Uzbek news classifier")
    parser.add_argument("--text-col", default=None, help="Text column name")
    parser.add_argument("--title-col", default=None, help="Title column name")
    parser.add_argument("--label-col", default=None, help="Label column name")
    parser.add_argument("--split", default=None, help="Dataset split name")
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--max-features", type=int, default=200000)
    parser.add_argument("--max-ngram", type=int, default=2)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=1e-5)
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hash-features", type=int, default=2**20)
    train(parser.parse_args())
