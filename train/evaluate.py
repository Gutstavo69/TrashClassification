"""Evaluate trained SVM on the held-out test split.

Usage (from project root):
    python -m train.evaluate

Outputs:
    models/confusion_matrix.png
    models/classification_report.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODELS_DIR = ROOT / "models"


def main() -> None:
    model = joblib.load(MODELS_DIR / "svm_model.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    le = joblib.load(MODELS_DIR / "label_encoder.joblib")
    split = joblib.load(MODELS_DIR / "test_split.joblib")
    X_test, y_test = split["X_test"], split["y_test"]

    X_test_s = scaler.transform(X_test)
    y_pred = model.predict(X_test_s)

    target_names = list(le.classes_)
    report = classification_report(
        y_test, y_pred, target_names=target_names, digits=4
    )
    print(report)
    (MODELS_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (test set)")
    fig.tight_layout()
    fig.savefig(MODELS_DIR / "confusion_matrix.png", dpi=150)
    print(f"Saved {MODELS_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
