"""Train SVM trash classifier on paper / glass / plastic.

Usage (from project root):
    python -m train.train

Outputs:
    models/svm_model.joblib
    models/scaler.joblib
    models/label_encoder.joblib
    models/test_split.joblib   (held-out X_test, y_test for evaluate.py)
    models/learning_curve.png
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV, learning_curve, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

# Make `src` importable when run as a script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.augmentation import augment_variants  # noqa: E402
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


def build_features(
    paths: list[str],
    labels: np.ndarray,
    augment: bool,
    base_seed: int = 0,
    log_prefix: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    feats: list[np.ndarray] = []
    out_labels: list[int] = []
    n = len(paths)
    t0 = time.time()
    for i, (p, lbl) in enumerate(zip(paths, labels)):
        bgr = load_bgr(str(p))
        feats.append(extract_all(bgr))
        out_labels.append(int(lbl))
        if augment:
            for aug in augment_variants(bgr, seed=base_seed + i):
                feats.append(extract_all(aug))
                out_labels.append(int(lbl))
        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"  {log_prefix}{i + 1}/{n}", flush=True)
    print(f"  {log_prefix}done in {time.time() - t0:.1f}s")
    return np.vstack(feats), np.asarray(out_labels)


def plot_learning_curve(estimator, X: np.ndarray, y: np.ndarray, save_path: Path) -> None:
    sizes, train_scores, val_scores = learning_curve(
        estimator,
        X,
        y,
        cv=5,
        train_sizes=np.linspace(0.1, 1.0, 6),
        scoring="f1_macro",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    val_mean, val_std = val_scores.mean(axis=1), val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sizes, train_mean, "o-", label="Training F1-macro")
    ax.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
    ax.plot(sizes, val_mean, "o-", label="CV F1-macro")
    ax.fill_between(sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("F1-macro")
    ax.set_title("Learning Curve (RBF SVM, 5-fold CV)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/6] Scanning dataset...")
    paths, labels = collect_paths()
    print(f"      total: {len(paths)} images")
    for cls in CLASSES:
        print(f"      {cls}: {labels.count(cls)}")

    le = LabelEncoder()
    y_all = le.fit_transform(labels)
    paths_arr = np.array([str(p) for p in paths])

    print("[2/6] Stratified split 70/15/15 on paths...")
    p_trainval, p_test, y_trainval, y_test_lbl = train_test_split(
        paths_arr, y_all, test_size=0.15, stratify=y_all, random_state=RANDOM_STATE
    )
    val_size_relative = 0.15 / 0.85
    p_train, p_val, y_train_lbl, y_val_lbl = train_test_split(
        p_trainval,
        y_trainval,
        test_size=val_size_relative,
        stratify=y_trainval,
        random_state=RANDOM_STATE,
    )
    print(f"      train paths={len(p_train)}  val={len(p_val)}  test={len(p_test)}")

    print("[3/6] Extracting features (train w/ augmentation, val/test plain)...")
    X_train, y_train = build_features(
        p_train, y_train_lbl, augment=True, log_prefix="train: "
    )
    X_val, y_val = build_features(p_val, y_val_lbl, augment=False, log_prefix="val: ")
    X_test, y_test = build_features(
        p_test, y_test_lbl, augment=False, log_prefix="test: "
    )
    print(
        f"      shapes: train={X_train.shape}  val={X_val.shape}  test={X_test.shape}"
    )

    print("[4/6] GridSearchCV (Pipeline: StandardScaler + RBF SVM, 5-fold)...")
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    probability=True,
                    decision_function_shape="ovo",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    param_grid = {
        "svm__C": [1, 10, 100],
        "svm__gamma": ["scale", 0.001, 0.01],
    }
    grid = GridSearchCV(
        pipe, param_grid, cv=5, scoring="f1_macro", n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)
    print(f"      best params: {grid.best_params_}")
    print(f"      best CV f1_macro: {grid.best_score_:.4f}")

    best_pipe: Pipeline = grid.best_estimator_
    val_acc = best_pipe.score(X_val, y_val)
    test_acc = best_pipe.score(X_test, y_test)
    print(f"      val acc:  {val_acc:.4f}")
    print(f"      test acc: {test_acc:.4f}")

    print("[5/6] Plotting learning curve (this may take several minutes)...")
    lc_path = MODELS_DIR / "learning_curve.png"
    plot_learning_curve(best_pipe, X_train, y_train, lc_path)
    print(f"      wrote {lc_path}")

    print("[6/6] Saving artifacts...")
    # Persist scaler and SVM separately for inference convenience
    fitted_scaler: StandardScaler = best_pipe.named_steps["scaler"]
    fitted_svm: SVC = best_pipe.named_steps["svm"]
    joblib.dump(fitted_svm, MODELS_DIR / "svm_model.joblib")
    joblib.dump(fitted_scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump(le, MODELS_DIR / "label_encoder.joblib")
    joblib.dump(
        {"X_test": X_test, "y_test": y_test},
        MODELS_DIR / "test_split.joblib",
    )
    print(f"      wrote files to {MODELS_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
