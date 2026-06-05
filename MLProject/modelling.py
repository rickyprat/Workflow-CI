"""
Kriteria 2 - Basic
Training model sentiment analysis Instagram menggunakan MLflow autolog.

Cara menjalankan dari folder Membangun_model:
1. pip install -r requirements.txt
2. python modelling.py
3. mlflow ui --backend-store-uri mlruns
4. Buka http://127.0.0.1:5000 lalu screenshot dashboard dan artifact.
"""

from pathlib import Path
import json
import warnings

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "instagram_sentiment_preprocessing"
TRAIN_PATH = DATA_DIR / "train_preprocessed.csv"
TEST_PATH = DATA_DIR / "test_preprocessed.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts_basic"
ARTIFACT_DIR.mkdir(exist_ok=True)

EXPERIMENT_NAME = "Instagram Sentiment Classification - Basic"
RANDOM_STATE = 42


def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    required_columns = {"clean_content", "label", "sentiment"}
    missing_train = required_columns - set(train_df.columns)
    missing_test = required_columns - set(test_df.columns)
    if missing_train or missing_test:
        raise ValueError(f"Kolom wajib tidak lengkap. train={missing_train}, test={missing_test}")

    X_train = train_df["clean_content"].fillna("").astype(str)
    y_train = train_df["label"].astype(int)
    X_test = test_df["clean_content"].fillna("").astype(str)
    y_test = test_df["label"].astype(int)
    return X_train, X_test, y_train, y_test


def main():
    X_train, X_test, y_train, y_test = load_data()

    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=3, max_df=0.9)),
            ("classifier", LinearSVC(random_state=RANDOM_STATE, C=1.0)),
        ]
    )

    mlflow.set_tracking_uri("file:" + str((BASE_DIR / "mlruns").resolve()))
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.sklearn.autolog(log_models=True)

    with mlflow.start_run(run_name="linear_svc_tfidf_autolog"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "test_recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "test_f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
        }

        # Autolog mencatat banyak parameter/model. Metrik test tetap dicatat eksplisit agar terlihat di UI.
        mlflow.log_metrics(metrics)

        report = classification_report(y_test, y_pred, target_names=["negatif", "netral", "positif"], zero_division=0)
        report_path = ARTIFACT_DIR / "classification_report_basic.txt"
        report_path.write_text(report, encoding="utf-8")
        mlflow.log_artifact(str(report_path), artifact_path="evaluation")

        model_path = ARTIFACT_DIR / "model_basic.joblib"
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="model_joblib")

        with open(ARTIFACT_DIR / "metrics_basic.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        mlflow.log_artifact(str(ARTIFACT_DIR / "metrics_basic.json"), artifact_path="evaluation")

        print("Training Basic selesai.")
        print(json.dumps(metrics, indent=4))
        print("Jalankan: mlflow ui --backend-store-uri mlruns")


if __name__ == "__main__":
    main()
