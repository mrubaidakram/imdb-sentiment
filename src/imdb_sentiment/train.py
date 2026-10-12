import argparse
import json
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from .data import ensure_data, load_split

def make_model():
    return Pipeline([("tfidf", TfidfVectorizer(strip_accents="unicode", lowercase=True, stop_words="english",
        ngram_range=(1, 2), min_df=2, max_features=150_000, sublinear_tf=True)),
        ("classifier", LogisticRegression(C=4.0, max_iter=1000, solver="liblinear", class_weight="balanced", random_state=42))])

def metrics(model, texts, labels):
    predictions = model.predict(texts)
    return {"accuracy": accuracy_score(labels, predictions), "macro_f1": f1_score(labels, predictions, average="macro"),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
        "classification_report": classification_report(labels, predictions, target_names=["negative", "positive"], output_dict=True, zero_division=0)}

def main():
    parser = argparse.ArgumentParser(description="Train a TF-IDF sentiment baseline on Stanford aclImdb.")
    parser.add_argument("--data-dir", default="data"); parser.add_argument("--output-dir", default="artifacts"); args = parser.parse_args()
    corpus = ensure_data(args.data_dir); train_text, train_y = load_split(corpus, "train"); test_text, test_y = load_split(corpus, "test")
    fit_text, val_text, fit_y, val_y = train_test_split(train_text, train_y, test_size=0.15, random_state=42, stratify=train_y)
    validation_model = make_model().fit(fit_text, fit_y); val_metrics = metrics(validation_model, val_text, val_y)
    final_model = make_model().fit(train_text, train_y); test_metrics = metrics(final_model, test_text, test_y)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, output / "sentiment_pipeline.joblib")
    report = {"validation": val_metrics, "test": test_metrics, "split": "official test set; validation drawn only from official training set", "seed": 42}
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"validation": val_metrics["accuracy"], "test": test_metrics["accuracy"], "macro_f1": test_metrics["macro_f1"]}, indent=2))

if __name__ == "__main__": main()
