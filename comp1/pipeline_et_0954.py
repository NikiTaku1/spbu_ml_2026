import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import make_scorer


# mape metric
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# loading data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

TARGET = "price_target"

y_raw = train_df[TARGET].values
y = np.log1p(y_raw)

test_ids = test_df["id"].values if "id" in test_df.columns else np.arange(len(test_df))

train_df = train_df.drop(columns=[TARGET])
test_df = test_df.drop(columns=["id"])

# preprocessing

cols_to_drop = [
        'location_logs_count_mean'
        'location_logs_count_std',
        'location_flash_mean_mean',
        'location_hds_ratio_mean_mean'
    ]
train_df = train_df.drop(columns=cols_to_drop, errors='ignore')
test_df = test_df.drop(columns=cols_to_drop, errors='ignore')


if 'rooms_4' in train_df.columns:
    train_df['rooms_4'] = train_df['rooms_4'].replace('1', 1)
    train_df['rooms_4'] = train_df['rooms_4'].replace('2', 2)
    train_df['rooms_4'] = train_df['rooms_4'].replace('3', 3)
    train_df['rooms_4'] = train_df['rooms_4'].replace('>=4', 4)
    train_df['rooms_4'] = train_df['rooms_4'].replace('студия', 0)
if 'rooms_4' in test_df.columns:
    test_df['rooms_4'] = test_df['rooms_4'].replace('1', 1)
    test_df['rooms_4'] = test_df['rooms_4'].replace('2', 2)
    test_df['rooms_4'] = test_df['rooms_4'].replace('3', 3)
    test_df['rooms_4'] = test_df['rooms_4'].replace('>=4', 4)
    test_df['rooms_4'] = test_df['rooms_4'].replace('студия', 0)
if 'agreement_date' in train_df.columns:
    train_df['agreement_date'] = pd.to_datetime(train_df['agreement_date'])
    train_df['year'] = train_df['agreement_date'].dt.year
    train_df['month'] = train_df['agreement_date'].dt.month
if 'agreement_date' in test_df.columns:
    test_df['agreement_date'] = pd.to_datetime(test_df['agreement_date'])
    test_df['year'] = test_df['agreement_date'].dt.year
    test_df['month'] = test_df['agreement_date'].dt.month

cat_cols = ['region_name_cat','district_cat','corpus_cat','developer_cat','agreement_date','hc_name_cat','interior_cat','class_cat','stage_cat']
print(cat_cols)

train_df = train_df.replace(-999, np.nan)
test_df = test_df.replace(-999, np.nan)

for c in train_df.columns:
    if c in cat_cols:
        train_df[c] = train_df[c].fillna("missing")
        test_df[c] = test_df[c].fillna("missing")
    else:
        med = train_df[c].median()
        train_df[c] = train_df[c].fillna(med)
        test_df[c] = test_df[c].fillna(med)

if 'floor' in train_df.columns:
    mask = (train_df['floor'] >= 1) & (train_df['floor'] <= 163)

    train_df = train_df[mask]
    y = y[mask]

train_le = train_df.copy()
test_le = test_df.copy()

for c in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train_le[c], test_le[c]]).astype(str)
    le.fit(combined)
    train_le[c] = le.transform(train_le[c].astype(str))
    test_le[c] = le.transform(test_le[c].astype(str))

X = train_le.values.astype(np.float32)
X_test = test_le.values.astype(np.float32)


# validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)


# optimisation
print("\nGridSearch for Extra Trees")

param_grid = {
    "n_estimators": [1800], #used to be more parameters, best chosen for compute speed
    "min_samples_split": [2],
    "min_samples_leaf": [1],
    "n_jobs": [-1],
    "random_state": [42]
}

# scorer (inverse because GridSearch maximizes)
mape_scorer = make_scorer(
    lambda yt, yp: -mape(np.expm1(yt), np.expm1(yp))
)

model = ExtraTreesRegressor()

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring=mape_scorer,
    cv=kf,
    verbose=2,
    n_jobs=-1
)

grid.fit(X, y)

best_params = grid.best_params_
best_score = -grid.best_score_

print("\n🏆 best:", best_score)
print("parameters:", best_params)


# retrain on all data
print("\nTraining final model\n")

final_model = ExtraTreesRegressor(**best_params)
final_model.fit(X, y)


# prediction
preds = np.expm1(final_model.predict(X_test))


# save result
submission = pd.DataFrame({
    "id": test_ids,
    "price_target": preds
})
submission.to_csv("et_submission.csv", index=False)

with open("et_params.txt", "w") as f:
    f.write(f"score: {best_score}\nparameters:\n{best_params}")

print("\nSaved")