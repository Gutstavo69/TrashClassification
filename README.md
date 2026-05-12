---
title: Trash Classifier SVM
emoji: ♻️
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
---

# Sistem Klasifikasi Sampah (Kertas, Kaca, Plastik) — SVM

SVM-based image classifier for 3 trash categories: **paper / glass / plastic**.
Features combine color histograms (RGB + HSV), LBP, GLCM, and HOG; classifier
is an RBF-kernel SVM tuned with 5-fold GridSearchCV.

This implementation follows the project proposal (`Comvis Proposal.pdf`):

| Proposal item                           | Where it lives                                  |
|-----------------------------------------|-------------------------------------------------|
| 224x224 resize                          | `train_model.ipynb` Section 3                   |
| Color histogram + LBP + GLCM + HOG      | `train_model.ipynb` Section 4                   |
| Augmentation (flip / rotate / zoom / brightness) | `train_model.ipynb` Section 5          |
| 70 / 15 / 15 stratified split           | `train_model.ipynb` Section 7                   |
| RBF SVM, OvO, GridSearchCV, 5-fold CV   | `train_model.ipynb` Section 9                   |
| Confusion matrix + per-class P/R/F1     | `train_model.ipynb` Section 10                  |
| Inference latency (target <=200 ms)     | `train_model.ipynb` Section 11                  |
| Accuracy & F1 learning curves           | `train_model.ipynb` Section 12                  |

## Files

```
.
├── app.py                  # Streamlit web app (loads the trained model)
├── train_model.ipynb       # Notebook: data -> features -> train -> evaluate -> save
├── predict_demo.ipynb      # Notebook: load model and try a single prediction
├── requirements.txt
├── Comvis Proposal.pdf
├── paper/  glass/  plastic/   # dataset folders (NOT committed)
└── models/                 # trained artifacts (committed)
    ├── svm_model.joblib
    ├── scaler.joblib
    ├── label_encoder.joblib
    ├── confusion_matrix.png
    ├── classification_report.txt
    ├── learning_curve_accuracy.png
    └── learning_curve_f1.png
```

The dataset folders (`paper/`, `glass/`, `plastic/`) live in the project root.
Download the TrashNet subset locally before training.

## How to run everything

### 0. One-time setup

```bash
pip install -r requirements.txt
```

You need Python 3.10+ and the dataset folders `paper/`, `glass/`, `plastic/`
sitting next to the notebooks (each containing `.jpg` images).

### 1. Train the model — `train_model.ipynb`

```bash
jupyter notebook train_model.ipynb
```

Then **Cell → Run All**. The notebook runs end to end:

1. Loads ~1577 images from `paper/ glass/ plastic/`
2. Stratified 70 / 15 / 15 split
3. Extracts features (with augmentation on the training set — this is the slow step, ~5-10 min)
4. GridSearchCV over `C ∈ {1, 10, 100}` × `gamma ∈ {scale, 0.001, 0.01}`, 5-fold CV, `f1_macro` scoring
5. Evaluates on the held-out test set (classification report + confusion matrix)
6. Measures inference latency on the test set and reports mean / median / p95 against the proposal's <=200 ms target
7. Plots accuracy and F1-macro learning curves
8. Saves the SVM, scaler, label encoder, and the test split to `models/`

When it finishes, `models/` will contain:

- `svm_model.joblib`, `scaler.joblib`, `label_encoder.joblib` — needed by `app.py`
- `test_split.joblib` — held-out features + labels
- `confusion_matrix.png`, `classification_report.txt`
- `learning_curve_accuracy.png`, `learning_curve_f1.png`

### 2. Quick prediction check — `predict_demo.ipynb`

```bash
jupyter notebook predict_demo.ipynb
```

Run all cells. The last cell uses `test_path = "paper/paper1.jpg"` — change it
to any image you want to classify. It prints the predicted class, confidence,
all class probabilities, and displays the image.

### 3. Run the web app — `app.py`

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Upload a JPG/PNG of trash and it shows:

- predicted class (Indonesian + English)
- confidence
- per-class probability bars
- per-image inference latency (the same metric as the notebook's Section 11)

The app will refuse to start with a clear error if `models/svm_model.joblib`
doesn't exist — you have to run `train_model.ipynb` first.

## Deploy to Hugging Face Spaces

1. Create a new Space (SDK: **Streamlit**).
2. Either link the GitHub repo (Space settings → "Sync from GitHub"),
   or push directly: `git remote add hf https://huggingface.co/spaces/<user>/<space> && git push hf main`.
3. HF picks up `app.py` and `requirements.txt` and builds automatically.

The trained `models/*.joblib` files must be committed for the Space to work —
training itself is not run on HF Spaces.

## References

Dataset: [TrashNet](https://github.com/garythung/trashnet) (Thung & Yang, Stanford 2016, MIT License).
