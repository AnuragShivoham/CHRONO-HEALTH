import json
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the SAME dataset used for training the XGBoost model
df = pd.read_csv("Data/processed/final_dataset_25k.csv")

# Remove disease columns
drop_cols = ["disease", "label"]
feature_names = [c for c in df.columns if c not in drop_cols]

print("Number of features:", len(feature_names))
print("Features:", feature_names)

# Fit scaler ONLY on the 46 symptom columns
scaler = StandardScaler()
scaler.fit(df[feature_names])

scaler_json = {
    "feature_names": feature_names,
    "mean": scaler.mean_.tolist(),
    "std": scaler.scale_.tolist()
}

with open("scaler.json", "w") as f:
    json.dump(scaler_json, f, indent=2)

print("scaler.json CREATED → length =", len(feature_names))
