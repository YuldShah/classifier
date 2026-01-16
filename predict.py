#!/usr/bin/env python3
import argparse
import os
import joblib


def predict(args: argparse.Namespace) -> None:
    model_obj = joblib.load(os.path.join(args.model_dir, "model.joblib"))
    label_encoder = joblib.load(os.path.join(args.model_dir, "label_encoder.joblib"))

    text = args.text.strip()

    if hasattr(model_obj, "predict"):
        pred = model_obj.predict([text])[0]
    else:
        vectorizer = model_obj["vectorizer"]
        clf = model_obj["classifier"]
        pred = clf.predict(vectorizer.transform([text]))[0]

    label = label_encoder.inverse_transform([pred])[0]

    print(label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Uzbek news category")
    parser.add_argument("--model-dir", default="artifacts")
    parser.add_argument("--text", required=True)
    predict(parser.parse_args())
