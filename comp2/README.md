# Model attempts

## 1st attempt
1st attempt: tf-idf + logistic regression

## 2nd attempt
2nd attempt: added Word2Vec (trained from scratch)

## 3rd attempt
3rd attempt: ensembled Log+TF-IDF and Log+TF-IDF+W2V

## 4th attempt
4th attempt: added LinearSVC (SVM) to ensemble

## 5th attempt
5th attempt: calibrated LinearSVC on TF-IDF

## 6th attempt
6th attempt: added soft-voting on ensemble weights

---

# Results

Best kaggle public score: 0.87804

Best kaggle private score: 0.87037 (actually 0.88571)

Tried but didn't improve score: xgb, lgbm on tf-idf

---

# Pipeline explanation

1. Read training and test datasets (text + labels).
2. Combine title + body, lowercase, remove URLs, symbols, and extra spaces.
3. Build TF-IDF features: word n-grams (1–2), character n-grams (2–4), then combine into one sparse feature matrix.
4. Train Word2Vec: Learn word embeddings from training text using skip-gram.
5. Convert each text into a vector using TF-IDF weighted average of Word2Vec vectors.
6. Merge TF-IDF (sparse) + Word2Vec embeddings (dense).
7. Create train/validation split (stratified).
8. Train models: Logistic Regression (TF-IDF), Logistic Regression (TF-IDF + W2V), Calibrated Linear SVM (TF-IDF)
9. Search best model weights, probability threshold
10. Train final models on full training data.
11. Combine model probabilities using best weights.
12. Save results