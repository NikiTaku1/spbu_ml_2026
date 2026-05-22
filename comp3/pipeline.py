import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error
from scipy.sparse import hstack, csr_matrix

from lightgbm import LGBMRegressor

# -----------------------
# Load data
# -----------------------
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# -----------------------
# Fill missing values
# -----------------------
text_cols = [
    'name_clean',
    'key_skills_name',
    'lemmaized_wo_stopwords_raw_description'
]

for col in text_cols:
    train[col] = train[col].fillna('')
    test[col] = test[col].fillna('')

cat_cols = [
    'experience_name',
    'schedule_name',
    'employment_name',
    'unified_address_city',
    'unified_address_state'
]

for col in cat_cols:
    train[col] = train[col].fillna('unknown')
    test[col] = test[col].fillna('unknown')

# -----------------------
# Numeric features
# -----------------------
def add_numeric_features(df):
    df = df.copy()
    
    df['title_len'] = df['name_clean'].str.len()
    df['skills_len'] = df['key_skills_name'].str.len()
    df['desc_len'] = df['lemmaized_wo_stopwords_raw_description'].str.len()
    
    df['title_word_count'] = df['name_clean'].str.split().apply(len)
    df['desc_word_count'] = df['lemmaized_wo_stopwords_raw_description'].str.split().apply(len)
    
    return df

train = add_numeric_features(train)
test = add_numeric_features(test)

# -----------------------
# Target
# -----------------------
y = np.log1p(train['salary_mean_net'])

# -----------------------
# Target Encoding
# -----------------------
def target_encode(train_df, test_df, col, target, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    train_encoded = np.zeros(len(train_df))
    test_encoded = np.zeros(len(test_df))
    
    global_mean = train_df[target].mean()
    
    for train_idx, val_idx in kf.split(train_df):
        tr, val = train_df.iloc[train_idx], train_df.iloc[val_idx]
        
        means = tr.groupby(col)[target].mean()
        
        train_encoded[val_idx] = val[col].map(means).fillna(global_mean)
    
    # test encoding
    means = train_df.groupby(col)[target].mean()
    test_encoded = test_df[col].map(means).fillna(global_mean)
    
    return np.log1p(train_encoded), np.log1p(test_encoded)

# Apply TE to multiple columns
te_cols = [
    'unified_address_city',
    'unified_address_state',
    'experience_name',
    'employment_name'
]

for col in te_cols:
    train[f'{col}_te'], test[f'{col}_te'] = target_encode(
        train, test, col, 'salary_mean_net'
    )

# -----------------------
# TF-IDF (data-driven text representation)
# -----------------------
tfidf_title = TfidfVectorizer(
    max_features=25000,
    ngram_range=(1,2),
    min_df=5
)

tfidf_desc = TfidfVectorizer(
    max_features=40000,
    ngram_range=(1,2),
    min_df=5
)

tfidf_skills = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,1),
    min_df=3
)

X_title_train = tfidf_title.fit_transform(train['name_clean'])
X_title_test = tfidf_title.transform(test['name_clean'])

X_desc_train = tfidf_desc.fit_transform(train['lemmaized_wo_stopwords_raw_description'])
X_desc_test = tfidf_desc.transform(test['lemmaized_wo_stopwords_raw_description'])

X_skills_train = tfidf_skills.fit_transform(train['key_skills_name'])
X_skills_test = tfidf_skills.transform(test['key_skills_name'])

# -----------------------
# Categorical
# -----------------------
ohe_cols = ['schedule_name']

ohe = OneHotEncoder(handle_unknown='ignore')

X_cat_train = ohe.fit_transform(train[ohe_cols])
X_cat_test = ohe.transform(test[ohe_cols])

# -----------------------
# Numeric features
# -----------------------
num_cols = [
    'title_len', 'skills_len', 'desc_len',
    'title_word_count', 'desc_word_count'
] + [f'{col}_te' for col in te_cols]

X_num_train = csr_matrix(train[num_cols].values)
X_num_test = csr_matrix(test[num_cols].values)

# -----------------------
# Final matrices
# -----------------------
X_train = hstack([
    X_title_train,
    X_desc_train,
    X_skills_train,
    X_cat_train,
    X_num_train
])

X_test = hstack([
    X_title_test,
    X_desc_test,
    X_skills_test,
    X_cat_test,
    X_num_test
])

# -----------------------
# Models
# -----------------------
lgb = LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=128,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

ridge = Ridge(alpha=4.0)

# -----------------------
# Cross-validation
# -----------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_oof = np.zeros(len(train))
ridge_oof = np.zeros(len(train))

test_lgb = np.zeros(len(test))
test_ridge = np.zeros(len(test))

for train_idx, val_idx in kf.split(X_train):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # LightGBM
    lgb.fit(X_tr, y_tr)
    lgb_oof[val_idx] = lgb.predict(X_val)
    test_lgb += lgb.predict(X_test) / 5
    
    # Ridge
    ridge.fit(X_tr, y_tr)
    ridge_oof[val_idx] = ridge.predict(X_val)
    test_ridge += ridge.predict(X_test) / 5

# -----------------------
# Ensemble
# -----------------------
oof = 0.7*lgb_oof + 0.3*ridge_oof
oof = np.expm1(oof)

y_true = np.expm1(y)

mape = mean_absolute_percentage_error(y_true, oof)
print("MAPE:", mape)

# -----------------------
# Final prediction
# -----------------------
final_pred = 0.7*test_lgb + 0.3*test_ridge
final_pred = np.expm1(final_pred)

# -----------------------
# Submission
# -----------------------
submission = pd.DataFrame({
    'id': test['id'],
    'salary_mean_net': final_pred
})

submission.to_csv("submission.csv", index=False)