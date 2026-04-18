import numpy as np
import pandas as pd
import optuna
import xgboost as xgb

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor, BayesianRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

target = "price_target"

y_raw = train_df[target].values
y = np.log1p(y_raw)

test_ids = test_df["id"].values if "id" in test_df.columns else np.arange(len(test_df))

train_df = train_df.drop(columns=[target])
test_df = test_df.drop(columns=["id"])

cat_cols = [c for c in train_df.columns if train_df[c].dtype == "object"]

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

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def tune_xgb():
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 5000, 8000),
            "max_depth": trial.suggest_int("max_depth", 5, 12),
            "learning_rate": trial.suggest_float("lr", 0.005, 0.02, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample", 0.5, 0.75),
            "reg_alpha": trial.suggest_float("alpha", 0, 0.5),
            "reg_lambda": trial.suggest_float("lambda", 0.2, 0.8),
            "n_jobs": -1,
            "verbosity": 0
        }

        oof = np.zeros(len(X))
        model = xgb.XGBRegressor(**params)

        for tr, va in kf.split(X):
            model.fit(X[tr], y[tr])
            oof[va] = model.predict(X[va])

        return mape(np.expm1(y), np.expm1(oof))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)

    return study.best_value, study.best_params

def cv_model(model_fn):
    oof = np.zeros(len(X))

    for tr, va in kf.split(X):
        model = model_fn()
        model.fit(X[tr], y[tr])
        oof[va] = model.predict(X[va])

    return mape(np.expm1(y), np.expm1(oof))

print("tuning xgboost")
xgb_score, xgb_params = tune_xgb()

print("evaluating random forest")
rf_score = cv_model(lambda: RandomForestRegressor(n_estimators=1000, n_jobs=-1, random_state=42))

print("evaluating extra trees")
et_score = cv_model(lambda: ExtraTreesRegressor(n_estimators=1000, n_jobs=-1, random_state=42))

print("evaluating ridge")
ridge_score = cv_model(lambda: Ridge(alpha=1.0))

print("evaluating lasso")
lasso_score = cv_model(lambda: Lasso(alpha=0.001))

print("evaluating elasticnet")
enet_score = cv_model(lambda: ElasticNet(alpha=0.001, l1_ratio=0.5))

print("evaluating svr")
svr_score = cv_model(lambda: SVR(C=10, epsilon=0.1))

print("evaluating knn")
knn_score = cv_model(lambda: KNeighborsRegressor(n_neighbors=10))

print("evaluating decision tree")
dt_score = cv_model(lambda: DecisionTreeRegressor(random_state=42))

print("evaluating huber")
huber_score = cv_model(lambda: HuberRegressor())

print("evaluating bayesian ridge")
br_score = cv_model(lambda: BayesianRidge())

scores = {
    "xgb": xgb_score,
    "rf": rf_score,
    "et": et_score,
    "ridge": ridge_score,
    "lasso": lasso_score,
    "enet": enet_score,
    "svr": svr_score,
    "knn": knn_score,
    "dt": dt_score,
    "huber": huber_score,
    "br": br_score
}

print("model scores")
for k, v in scores.items():
    print(k, v)

best_model_name = min(scores, key=scores.get)

print("best model", best_model_name)

print("training final model")

if best_model_name == "xgb":
    final_model = xgb.XGBRegressor(**xgb_params)
elif best_model_name == "rf":
    final_model = RandomForestRegressor(n_estimators=1000, n_jobs=-1, random_state=42)
elif best_model_name == "et":
    final_model = ExtraTreesRegressor(n_estimators=1000, n_jobs=-1, random_state=42)
elif best_model_name == "ridge":
    final_model = Ridge(alpha=1.0)
elif best_model_name == "lasso":
    final_model = Lasso(alpha=0.001)
elif best_model_name == "enet":
    final_model = ElasticNet(alpha=0.001, l1_ratio=0.5)
elif best_model_name == "svr":
    final_model = SVR(C=10, epsilon=0.1)
elif best_model_name == "knn":
    final_model = KNeighborsRegressor(n_neighbors=10)
elif best_model_name == "dt":
    final_model = DecisionTreeRegressor(random_state=42)
elif best_model_name == "huber":
    final_model = HuberRegressor()
else:
    final_model = BayesianRidge()

final_model.fit(X, y)

test_preds_log = final_model.predict(X_test)
test_preds = np.expm1(test_preds_log)

submission = pd.DataFrame({
    "id": test_ids,
    "price_target": test_preds
})

submission.to_csv("final_submission.csv", index=False)

print("saved final_submission.csv")

with open("model_params.txt", "w", encoding="utf-8") as f:
    f.write("model scores\n\n")

    f.write(f"xgboost score: {xgb_score}\n")
    f.write(f"xgboost params: {xgb_params}\n\n")

    f.write(f"random forest score: {rf_score}\n")
    f.write("rf params: n_estimators=1000, n_jobs=-1, random_state=42\n\n")

    f.write(f"extra trees score: {et_score}\n")
    f.write("et params: n_estimators=1000, n_jobs=-1, random_state=42\n\n")

    f.write(f"ridge score: {ridge_score}\n")
    f.write("ridge params: alpha=1.0\n\n")

    f.write(f"lasso score: {lasso_score}\n")
    f.write("lasso params: alpha=0.001\n\n")

    f.write(f"elasticnet score: {enet_score}\n")
    f.write("elasticnet params: alpha=0.001, l1_ratio=0.5\n\n")

    f.write(f"svr score: {svr_score}\n")
    f.write("svr params: C=10, epsilon=0.1\n\n")

    f.write(f"knn score: {knn_score}\n")
    f.write("knn params: n_neighbors=10\n\n")

    f.write(f"decision tree score: {dt_score}\n")
    f.write("dt params: random_state=42\n\n")

    f.write(f"huber score: {huber_score}\n")
    f.write("huber params: default\n\n")

    f.write(f"bayesian ridge score: {br_score}\n")
    f.write("br params: default\n\n")

    f.write(f"best model: {best_model_name}\n\n")

    f.write("final model config\n")

    if best_model_name == "xgb":
        f.write("model: xgboost\n")
        f.write(str(xgb_params))
    elif best_model_name == "rf":
        f.write("model: random forest\n")
        f.write("n_estimators=1000, n_jobs=-1, random_state=42")
    elif best_model_name == "et":
        f.write("model: extra trees\n")
        f.write("n_estimators=1000, n_jobs=-1, random_state=42")
    elif best_model_name == "ridge":
        f.write("model: ridge\nalpha=1.0")
    elif best_model_name == "lasso":
        f.write("model: lasso\nalpha=0.001")
    elif best_model_name == "enet":
        f.write("model: elasticnet\nalpha=0.001, l1_ratio=0.5")
    elif best_model_name == "svr":
        f.write("model: svr\nC=10, epsilon=0.1")
    elif best_model_name == "knn":
        f.write("model: knn\nn_neighbors=10")
    elif best_model_name == "dt":
        f.write("model: decision tree\nrandom_state=42")
    elif best_model_name == "huber":
        f.write("model: huber\ndefault")
    else:
        f.write("model: bayesian ridge\ndefault")

print("saved model_params.txt")