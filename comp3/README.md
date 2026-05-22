# Model attempts

## 1st attempt
1st attempt: tf-idf on text features + logistic regression

## 2nd attempt
2nd attempt: tf-idf on text features + lgbm/ridge ensemble

## 3rd attempt
3rd attempt: added target log

## 4th attempt
4th attempt: added target encoding for certain variables

## 5th attempt
5th attempt: different ngrams for different text variables

---

# Results

Best kaggle public score: 0.20612

Best kaggle private score: 0.20682

Tried but didn't improve score: catboost, xgb, other tree-based models, dropping/shortening text entries

---

# Pipeline explanation

1. Read training and test datasets containing job text, categorical features, and salary targets.
2. Fill missing text fields with empty strings and missing categorical fields with "unknown".
3. Create numeric text features such as title length, description length, and word counts.
4. Apply log transformation to the target salary (log1p).
5. Build target encoding features for city, state, experience, and employment type using 5-fold cross-validation.
6. Generate TF-IDF features for job titles using word uni-grams and bi-grams.
7. Generate TF-IDF features for job descriptions using word uni-grams and bi-grams.
8. Generate TF-IDF features for key skills using uni-grams.
9. One-hot encode categorical schedule features.
10. Combine TF-IDF features, one-hot encoded categories, numeric features, and target-encoded features into one sparse feature matrix.
11. Create 5-fold cross-validation splits.
12. Train LightGBM regression model on each fold and generate out-of-fold/test predictions.
13. Train Ridge Regression model on each fold and generate out-of-fold/test predictions.
14. Ensemble predictions using weighted averaging (70% LightGBM + 30% Ridge).
15. Apply inverse log transformation to convert predictions back to salary scale.
16. Evaluate performance using MAPE.
17. Train final ensemble predictions for the test dataset.
18. Save final predictions.