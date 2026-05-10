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
Features combine color histograms (RGB + HSV), LBP, GLCM, and HOG; classifier is
an RBF-kernel SVM tuned with 5-fold GridSearchCV.

## Project layout

```
.
├── app.py                # Streamlit app (HF Spaces entrypoint)
├── requirements.txt
├── src/
│   ├── preprocessing.py  # resize / color conversion
│   ├── features.py       # extract_all() — shared by training & inference
│   └── inference.py      # Classifier wrapper
├── train/
│   ├── train.py          # full training pipeline
│   └── evaluate.py       # confusion matrix + classification report
└── models/               # trained artifacts (committed)
    ├── svm_model.joblib
    ├── scaler.joblib
    └── label_encoder.joblib
```

The dataset folders (`paper/`, `glass/`, `plastic/`) are **not committed** —
download the TrashNet subset locally before training.

## Quickstart

```bash
pip install -r requirements.txt

# Place dataset at ./paper/, ./glass/, ./plastic/, then:
python -m train.train
python -m train.evaluate

# Run the website locally:
streamlit run app.py
```

## Deploy to Hugging Face Spaces

1. Create a new Space (SDK: **Streamlit**).
2. Either link the GitHub repo (Space settings → "Sync from GitHub"),
   or push directly: `git remote add hf https://huggingface.co/spaces/<user>/<space> && git push hf main`.
3. HF picks up `app.py` and `requirements.txt` and builds automatically.

The trained `models/*.joblib` files must be committed for the Space to work —
training itself is not run on HF Spaces.

## References

Dataset: [TrashNet](https://github.com/garythung/trashnet) (Thung & Yang, Stanford 2016, MIT License).git init