import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

DATA = "Data/processed/final_dataset_25k.csv"
OUT_DIR = "Data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA)
target = "disease_encoded"
features = [c for c in df.columns if c not in ["disease", target]]

X = df[features].values
y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# RandomForest
rf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("RF Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))
joblib.dump(rf, f"{OUT_DIR}/rf_25k.pkl")

# XGBoost
xgb = XGBClassifier(
    n_estimators=600,
    max_depth=7,
    learning_rate=0.05,
    eval_metric="mlogloss",
    n_jobs=-1
)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
print("XGB Accuracy:", accuracy_score(y_test, xgb_pred))
print(classification_report(y_test, xgb_pred))
joblib.dump(xgb, f"{OUT_DIR}/xgb_25k.pkl")

# Save feature list
with open(f"{OUT_DIR}/feature_list_25k.txt","w") as f:
    for name in features:
        f.write(name+"\n")

print("DONE.")
