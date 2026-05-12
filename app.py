# Trash classifier Streamlit app.
# Loads the SVM trained in train_model.ipynb and predicts on uploaded images.

import io
import os
import time

import cv2
import joblib
import numpy as np
import streamlit as st
from PIL import Image
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops, hog


MODELS_DIR = "models"
IMG_SIZE = 224
LABELS_ID = {"paper": "Kertas", "glass": "Kaca", "plastic": "Plastik"}


st.set_page_config(
    page_title="Trash Classifier (SVM)",
    page_icon="♻️",
    layout="centered",
)


# ---- feature extraction (must match what the notebook trained with) ----

def resize_img(img):
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE))


def color_hist(img):
    feats = []
    for c in range(3):
        h = cv2.calcHist([img], [c], None, [16], [0, 256]).flatten()
        h = h / (h.sum() + 1e-7)
        feats.append(h)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ranges = [(0, 180), (0, 256), (0, 256)]
    for c, r in enumerate(ranges):
        h = cv2.calcHist([hsv], [c], None, [16], list(r)).flatten()
        h = h / (h.sum() + 1e-7)
        feats.append(h)
    return np.concatenate(feats)


def lbp_hist(gray):
    lbp = local_binary_pattern(gray, 24, 3, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26))
    hist = hist.astype("float32")
    hist /= (hist.sum() + 1e-7)
    return hist


def glcm_feat(gray):
    g = (gray // 32).astype(np.uint8)
    glcm = graycomatrix(
        g,
        distances=[1, 2],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=8,
        symmetric=True,
        normed=True,
    )
    out = []
    for p in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
        out.append(graycoprops(glcm, p).flatten())
    return np.concatenate(out).astype("float32")


def hog_feat(gray):
    return hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype("float32")


def extract_features(img):
    img = resize_img(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.concatenate([
        color_hist(img),
        lbp_hist(gray),
        glcm_feat(gray),
        hog_feat(gray),
    ]).astype("float32")


# ---- model loading + prediction ----

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    svm = joblib.load(os.path.join(MODELS_DIR, "svm_model.joblib"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.joblib"))
    return svm, scaler, le


def predict(img_bgr, svm, scaler, le):
    feats = extract_features(img_bgr).reshape(1, -1)
    feats_s = scaler.transform(feats)
    probs = svm.predict_proba(feats_s)[0]
    idx = int(np.argmax(probs))
    label = str(le.classes_[idx])
    prob_map = {str(c): float(probs[i]) for i, c in enumerate(le.classes_)}
    return label, float(probs[idx]), prob_map


def pil_to_bgr(img):
    rgb = np.array(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ---- streamlit UI ----

def main():
    st.title("♻️ Trash Classifier")
    st.caption(
        "SVM (RBF) — Paper / Glass / Plastic. "
        "Features: color histogram + LBP + GLCM + HOG."
    )

    if not os.path.exists(os.path.join(MODELS_DIR, "svm_model.joblib")):
        st.error(
            "Model files not found in `models/`. "
            "Run `train_model.ipynb` first, then commit "
            "`models/svm_model.joblib`, `models/scaler.joblib`, "
            "and `models/label_encoder.joblib`."
        )
        return

    svm, scaler, le = load_model()

    uploaded = st.file_uploader(
        "Upload an image of trash (jpg / png)",
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
        label, conf, probs = predict(bgr, svm, scaler, le)
        latency_ms = (time.perf_counter() - t0) * 1000

        label_id = LABELS_ID.get(label, label)
        st.metric(label="Prediction", value=f"{label_id} ({label})")
        st.metric(label="Confidence", value=f"{conf * 100:.1f}%")
        st.caption(f"Inference time: {latency_ms:.1f} ms")

    st.subheader("Class probabilities")
    sorted_probs = dict(sorted(probs.items(), key=lambda kv: kv[1], reverse=True))
    for c, p in sorted_probs.items():
        st.write(f"**{LABELS_ID.get(c, c)}** ({c})")
        st.progress(float(p), text=f"{p * 100:.1f}%")


if __name__ == "__main__":
    main()
