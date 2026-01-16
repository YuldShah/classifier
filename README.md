# Uzbek News Classifier

## Setup

1. Install dependencies:

- `pip install -r requirements.txt`

## Train

- `python train.py`

Optional arguments:

- `--text-col text`
- `--title-col title`
- `--label-col folder`
- `--split train`

### Streaming mode (low memory)

If you want to avoid loading the full dataset into memory, enable streaming and set limits:

- `python train.py --streaming --max-train 400000 --max-val 20000`

Artifacts are saved in the `artifacts/` folder.

## Predict

- `python predict.py --text "your news text here"`

## Transformer fine-tune

Install dependencies:

- `pip install -r requirements.txt`

Train:

- `python train_transformer.py --model-name xlm-roberta-base --max-samples 50000 --epochs 2`

Resume training from a checkpoint and continue on a different slice:

- `python train_transformer.py --model-name xlm-roberta-base --start 100000 --max-samples 400000 --val-size 0.125 --shuffle --resume-from-checkpoint artifacts_transformer/checkpoint-XXXX`

Predict:

- `python predict_transformer.py --text "your news text here"`
