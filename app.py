import io
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.inference import Classifier

MODELS_DIR = Path(__file__).resolve().parent / "models"
LABELS_ID = {"paper": "Kertas", "glass": "Kaca", "plastic": "Plastik"}

st.set_page_config(
    page_title="Trash Classifier (SVM)",
    page_icon="♻️",
    layout="centered",
)


@st.cache_resource(show_spinner="Loading model...")
def get_classifier() -> Classifier:
    return Classifier(MODELS_DIR)


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    rgb = np.array(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def main() -> None:
    st.title("♻️ Trash Classifier")
    st.caption(
        "SVM (RBF) — Paper / Glass / Plastic. "
        "Features: color histogram + LBP + GLCM + HOG."
    )

    if not (MODELS_DIR / "svm_model.joblib").exists():
        st.error(
            "Model artifacts not found in `models/`. "
            "Run `python -m train.train` locally first, then commit "
            "`models/svm_model.joblib`, `models/scaler.joblib`, "
            "and `models/label_encoder.joblib`."
        )
        return

    clf = get_classifier()

    uploaded = st.file_uploader(
        "Upload an image of trash (jpg/png)",
        type=["jpg", "jpeg", "png"],
    )
    if uploaded is None:
        st.info("Upload an image to classify it.")
        return

    img = Image.open(io.BytesIO(uploaded.read()))
    bgr = pil_to_bgr(img)

    col_img, col_pred = st.columns([1, 1])
    with col_img:
        st.image(img, caption=uploaded.name, use_column_width=True)

    with col_pred:
        t0 = time.perf_counter()
        pred = clf.predict(bgr)
        latency_ms = (time.perf_counter() - t0) * 1000

        label_id = LABELS_ID.get(pred.label, pred.label)
        st.metric(
            label="Prediction",
            value=f"{label_id} ({pred.label})",
        )
        st.metric(
            label="Confidence",
            value=f"{pred.confidence * 100:.1f}%",
        )
        st.caption(f"Inference time: {latency_ms:.1f} ms")

    st.subheader("Class probabilities")
    sorted_probs = dict(
        sorted(pred.probabilities.items(), key=lambda kv: kv[1], reverse=True)
    )
    for cls, p in sorted_probs.items():
        st.write(f"**{LABELS_ID.get(cls, cls)}** ({cls})")
        st.progress(float(p), text=f"{p * 100:.1f}%")


if __name__ == "__main__":
    main()
