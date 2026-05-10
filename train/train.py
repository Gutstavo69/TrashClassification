"""Train SVM trash classifier on paper / glass / plastic.

Usage (from project root):
    python -m train.train

Outputs:
    models/svm_model.joblib
    models/scaler.joblib
    models/label_encoder.joblib
    models/test_split.joblib   (held-out X_test, y_test for evaluate.py)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

# Make `src` importable when run as a script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import extract_all  # noqa: E402
from src.preprocessing import load_bgr  # noqa: E402

CLASSES = ("paper", "glass", "plastic")
DATA_ROOT = ROOT
MODELS_DIR = ROOT / "models"
RANDOM_STATE = 42


def collect_paths() -> tuple[list[Path], list[str]]:
    paths: list[Path] = []
    labels: list[str] = []
    for cls in CLASSES:
        cls_dir = DATA_ROOT / cls
        if not cls_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {cls_dir}")
        files = sorted(p for p in cls_dir.iterdir() if p.suffix.lower() == ".jpg")
        if not files:
            raise FileNotFoundError(f"No .jpg files in {cls_dir}")
        paths.extend(files)
        labels.extend([cls] * len(files))
    return paths, labels


def build_feature_matrix(paths: list[Path]) -> np.ndarray:
    feats = []
    n = len(paths)
    for i, p in enumerate(paths, 1):
        bgr = load_bgr(str(p))
        feats.append(extract_all(bgr))
        if i % 50 == 0 or i == n:
            print(f"  features: {i}/{n}", flush=True)
    return np.vstack(feats)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/5] Scanning dataset...")
    paths, labels = collect_paths()
    print(f"      total: {len(paths)} images")
    for cls in CLASSES:
        print(f"      {cls}: {labels.count(cls)}")

    le = LabelEncoder()
    y = le.fit_transform(labels)

    print("[2/5] Extracting features...")
    t0 = time.time()
    X = build_feature_matrix(paths)
    print(f"      X shape: {X.shape}  (took {time.time() - t0:.1f}s)")

    print("[3/5] Stratified split 70/15/15...")
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE
    )
    val_size_relative = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_size_relative,
        stratify=y_trainval,
        random_state=RANDOM_STATE,
    )
    print(
        f"      train={len(y_train)}  val={len(y_val)}  test={len(y_test)}"
    )

    print("[4/5] Scaling + GridSearchCV (RBF SVM, 5-fold)...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    param_grid = {
        "C": [1, 10, 100],
        "gamma": ["scale", 0.001, 0.01],
    }
    base = SVC(kernel="rbf", probability=True, decision_function_shape="ovo",
               random_state=RANDOM_STATE)
    grid = GridSearchCV(
        base, param_grid, cv=5, scoring="f1_macro", n_jobs=-1, verbose=1
    )
    grid.fit(X_train_s, y_train)
    print(f"      best params: {grid.best_params_}")
    print(f"      best CV f1_macro: {grid.best_score_:.4f}")

    model = grid.best_estimator_
    val_acc = model.score(X_val_s, y_val)
    test_acc = model.score(X_test_s, y_test)
    print(f"      val acc:  {val_acc:.4f}")
    print(f"      test acc: {test_acc:.4f}")

    print("[5/5] Saving artifacts...")
    joblib.dump(model, MODELS_DIR / "svm_model.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump(le, MODELS_DIR / "label_encoder.joblib")
    joblib.dump(
        {"X_test": X_test, "y_test": y_test},
        MODELS_DIR / "test_split.joblib",
    )
    print(f"      wrote files to {MODELS_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
