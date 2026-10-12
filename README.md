# IMDb Sentiment Analysis

> A transparent text-classification baseline using the Stanford Large Movie Review Dataset, TF-IDF, and logistic regression.

## Goal

Classical NLP benchmark with a train-only validation split, held-out test evaluation, macro-F1, and confusion matrix. A compact baseline for later transformer comparisons.

## Dataset

Stanford Large Movie Review Dataset (aclImdb): 25,000 labeled training reviews, 25,000 labeled test reviews, and 50,000 unlabeled reviews. The official archive is downloaded on first run and extracted safely into `data/`.

**Official source:** [https://ai.stanford.edu/~amaas/data/sentiment/](https://ai.stanford.edu/~amaas/data/sentiment/)
**Data terms:** Follow the dataset README and original paper citation. The review archive is not bundled or redistributed by this repository.

Raw dataset files are not committed. First run downloads data into the ignored `data/` directory.

## Run

Python 3.10 or newer is required. Create a virtual environment, install the local package, and train:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
python -m imdb_sentiment.train --data-dir data --output-dir artifacts
```

The first run may download the dataset. Checkpoint and metrics output is written to `artifacts/`.

## Reproducibility and evaluation

- Training, validation, and test responsibilities are documented in the training code.
- Random seeds and key hyperparameters are explicit CLI options or constants.
- Test metrics are reported only after model selection on validation data.
- Results are not published in this README until a reproducible run has produced them.

## Stack

NLP · text classification · TF-IDF · scikit-learn · evaluation

## Scope

This is an independently built portfolio project using public benchmark data. It contains no client records, employer code, or claims that it reproduces private production systems. Dataset attribution and license obligations remain with users of the data.

## Validation status

Python source syntax has been checked. Dependencies have not been installed and training has not been run; runtime behavior and model quality are not yet verified.
