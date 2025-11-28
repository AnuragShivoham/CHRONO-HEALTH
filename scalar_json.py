import json
import pandas as pd
from sklearn.preprocessing import StandardScaler

print("Loading dataset...")

df = pd.read_csv("Data/processed/clean_dataset.csv")

print("Dataset loaded:", df.shape)

# Disease/label columns are removed
drop_cols = ["disease", "label"]
feature_names = [c for c in df.columns if c not in drop_cols]

print("Feature count:", len(feature_names))
print("Feature names:", feature_names)

scaler = StandardScaler()
scaler.fit(df[feature_names])

scaler_json = {
    "feature_names": feature_names,
    "mean": scaler.mean_.tolist(),
    "std": scaler.scale_.tolist()
}

with open("scaler.json", "w") as f:
    json.dump(scaler_json, f, indent=2)

print("scaler.json created successfully!")
