"""
Depression Detection – Binary Classification Pipeline
======================================================
Models used:
  - TF-IDF word n-grams (1-2) + char n-grams (2-4)
  - Word2Vec trained FROM SCRATCH via skip-gram + negative sampling (gensim)
  - TF-IDF weighted document embeddings from Word2Vec
  - Logistic Regression (on TF-IDF only, and TF-IDF + W2V embeddings)
  - LinearSVC (calibrated) on TF-IDF
  - Soft-voting ensemble with weight + threshold optimisation

Usage:
    pip install scikit-learn gensim
    python pipeline.py
    # outputs submission.csv
"""

import pandas as pd
import numpy as np
import re
import warnings
import scipy.sparse as sp

warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score
from gensim.models import Word2Vec

np.random.seed(42)

# ── CONFIG ────────────────────────────────────────────────────────────────────
TRAIN_PATH = "train.csv"
TEST_PATH  = "test.csv"
OUT_PATH   = "submission.csv"

W2V_DIM     = 100
W2V_WINDOW  = 5
W2V_MIN_CNT = 3
W2V_EPOCHS  = 10
W2V_NEG     = 5
CLASS_WEIGHT = {0: 1, 1: 4}   # handle ~4:1 imbalance


# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
print(f"Train: {train.shape}  Test: {test.shape}")
print("Label distribution:", train["label"].value_counts().to_dict())


# ── 2. TEXT CLEANING ──────────────────────────────────────────────────────────
def clean(title, body):
    t = str(title) if pd.notna(title) else ""
    b = str(body)  if pd.notna(body)  else ""
    text = (t + " " + b).lower()
    text = re.sub(r"http\S+", " ", text)                     # remove URLs
    text = re.sub(r"\[removed\]|\[deleted\]", " ", text, flags=re.I)
    text = re.sub(r"[^a-z0-9\s]", " ", text)                # keep alphanum
    return re.sub(r"\s+", " ", text).strip()

train["text"] = [clean(t, b) for t, b in zip(train["title"], train["body"])]
test["text"]  = [clean(t, b) for t, b in zip(test["title"],  test["body"])]

X_tr = train["text"].values
X_te = test["text"].values
y_tr = train["label"].values.astype(int)


# ── 3. TF-IDF FEATURES ────────────────────────────────────────────────────────
print("\nBuilding TF-IDF features...")

tfidf_word = TfidfVectorizer(
    max_features=40_000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=3,
    strip_accents="unicode",
    analyzer="word",
)
tfidf_char = TfidfVectorizer(
    max_features=20_000,
    ngram_range=(2, 4),
    sublinear_tf=True,
    min_df=4,
    strip_accents="unicode",
    analyzer="char_wb",
)

Tw_tr = tfidf_word.fit_transform(X_tr);  Tw_te = tfidf_word.transform(X_te)
Tc_tr = tfidf_char.fit_transform(X_tr);  Tc_te = tfidf_char.transform(X_te)

S_tr = sp.hstack([Tw_tr, Tc_tr])   # sparse-only feature matrix
S_te = sp.hstack([Tw_te, Tc_te])
print(f"  Sparse TF-IDF shape: {S_tr.shape}")


# ── 4. WORD2VEC (trained from scratch) ───────────────────────────────────────
print("\nTraining Word2Vec from scratch (skip-gram + negative sampling)...")

sentences = [s.split() for s in X_tr]
w2v = Word2Vec(
    sentences=sentences,
    vector_size=W2V_DIM,
    window=W2V_WINDOW,
    min_count=W2V_MIN_CNT,
    sg=1,               # skip-gram
    negative=W2V_NEG,   # negative sampling
    epochs=W2V_EPOCHS,
    workers=4,
    seed=42,
)
print(f"  Vocabulary size: {len(w2v.wv)}")


# ── 5. DOCUMENT EMBEDDINGS (TF-IDF weighted mean pooling) ────────────────────
idf_map = dict(zip(tfidf_word.get_feature_names_out(), tfidf_word.idf_))

def doc_embed(text):
    """TF-IDF weighted average of Word2Vec token vectors."""
    tokens = text.split()
    vecs, weights = [], []
    for tok in tokens:
        if tok in w2v.wv:
            vecs.append(w2v.wv[tok])
            weights.append(idf_map.get(tok, 1.0))
    if vecs:
        weights = np.array(weights)
        weights /= weights.sum()
        return np.average(vecs, axis=0, weights=weights)
    return np.zeros(W2V_DIM)

print("Computing document embeddings...")
E_tr = np.array([doc_embed(t) for t in X_tr])
E_te = np.array([doc_embed(t) for t in X_te])

# Full feature matrix: TF-IDF sparse + dense W2V embeddings
F_tr = sp.hstack([S_tr, sp.csr_matrix(E_tr)])
F_te = sp.hstack([S_te, sp.csr_matrix(E_te)])
print(f"  Full feature shape: {F_tr.shape}")


# ── 6. TRAIN / VAL SPLIT FOR TUNING ──────────────────────────────────────────
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, val_idx = next(sss.split(np.arange(len(y_tr)), y_tr))


# ── 7. FIT CLASSIFIERS ────────────────────────────────────────────────────────
print("\nFitting classifiers...")

# Model A – Logistic Regression on TF-IDF
lra = LogisticRegression(C=0.5, class_weight=CLASS_WEIGHT,
                         max_iter=500, solver="lbfgs")
lra.fit(S_tr[tr_idx], y_tr[tr_idx])
print("  LR (TF-IDF) done")

# Model B – Logistic Regression on TF-IDF + W2V
lrb = LogisticRegression(C=0.3, class_weight=CLASS_WEIGHT,
                         max_iter=500, solver="lbfgs")
lrb.fit(F_tr[tr_idx], y_tr[tr_idx])
print("  LR (TF-IDF + W2V) done")

# Model C – LinearSVC (calibrated) on TF-IDF
svc_val = CalibratedClassifierCV(
    LinearSVC(C=0.3, class_weight=CLASS_WEIGHT, max_iter=2000), cv=3)
svc_val.fit(S_tr[tr_idx], y_tr[tr_idx])
print("  LinearSVC (calibrated) done")

# Also fit SVC on full training set for final predictions
svc_full = CalibratedClassifierCV(
    LinearSVC(C=0.3, class_weight=CLASS_WEIGHT, max_iter=2000), cv=3)
svc_full.fit(S_tr, y_tr)


# ── 8. ENSEMBLE WEIGHT + THRESHOLD SEARCH (on val split) ─────────────────────
pa = lra.predict_proba(S_tr[val_idx])[:, 1]
pb = lrb.predict_proba(F_tr[val_idx])[:, 1]
pc = svc_val.predict_proba(S_tr[val_idx])[:, 1]
y_val = y_tr[val_idx]

best_f1, best_cfg = 0.0, (0.2, 0.6, 0.2, 0.5)
for w1 in np.linspace(0.1, 0.8, 8):
    for w2 in np.linspace(0.1, 0.8, 8):
        w3 = max(1.0 - w1 - w2, 0.0)
        if w3 < 0.05:
            continue
        p = w1 * pa + w2 * pb + w3 * pc
        for th in np.arange(0.20, 0.80, 0.02):
            f1 = f1_score(y_val, (p > th).astype(int))
            if f1 > best_f1:
                best_f1, best_cfg = f1, (w1, w2, w3, th)

w1, w2, w3, threshold = best_cfg
print(f"\nBest val F1 : {best_f1:.4f}")
print(f"Weights     : LR_tfidf={w1:.2f}, LR_full={w2:.2f}, SVC={w3:.2f}")
print(f"Threshold   : {threshold:.2f}")


# ── 9. RETRAIN ON FULL DATA + FINAL PREDICTIONS ───────────────────────────────
print("\nRetraining on full training set...")

lra2 = LogisticRegression(C=0.5, class_weight=CLASS_WEIGHT,
                          max_iter=500, solver="lbfgs").fit(S_tr, y_tr)
lrb2 = LogisticRegression(C=0.3, class_weight=CLASS_WEIGHT,
                          max_iter=500, solver="lbfgs").fit(F_tr, y_tr)

pa2 = lra2.predict_proba(S_te)[:, 1]
pb2 = lrb2.predict_proba(F_te)[:, 1]
pc2 = svc_full.predict_proba(S_te)[:, 1]

final_proba = w1 * pa2 + w2 * pb2 + w3 * pc2
final_preds = (final_proba > threshold).astype(int)


# ── 10. SAVE SUBMISSION ───────────────────────────────────────────────────────
submission = pd.DataFrame({"id": test["id"], "label": final_preds})
submission.to_csv(OUT_PATH, index=False)
print(f"\nSubmission saved to {OUT_PATH}")
print("Prediction distribution:", submission["label"].value_counts().to_dict())
print(submission.head(10).to_string(index=False))
