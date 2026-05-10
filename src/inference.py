from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from .features import extract_all

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


@dataclass
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


class Classifier:
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.model = joblib.load(models_dir / "svm_model.joblib")
        self.scaler = joblib.load(models_dir / "scaler.joblib")
        self.label_encoder = joblib.load(models_dir / "label_encoder.joblib")

    def predict(self, bgr: np.ndarray) -> Prediction:
        feats = extract_all(bgr).reshape(1, -1)
        feats_s = self.scaler.transform(feats)
        probs = self.model.predict_proba(feats_s)[0]
        idx = int(np.argmax(probs))
        label = str(self.label_encoder.classes_[idx])
        prob_map = {
            str(cls): float(probs[i])
            for i, cls in enumerate(self.label_encoder.classes_)
        }
        return Prediction(
            label=label,
            confidence=float(probs[idx]),
            probabilities=prob_map,
        )
