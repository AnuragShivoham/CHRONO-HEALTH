import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import joblib

IN = "Data/interim/prepared_25k.csv"
OUT = "Data/processed/final_dataset_25k.csv"
SCALER_OUT = "Data/processed/scaler_25k.pkl"

os.makedirs("Data/processed", exist_ok=True)

df = pd.read_csv(IN)

target = "disease_encoded"
exclude = ["disease", target]
features = [c for c in df.columns if c not in exclude]

# Scale only vital signs
scale_cols = ["age","heart_rate","blood_pressure","cholesterol_level"]
scaler = StandardScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])

joblib.dump(scaler, SCALER_OUT)
print("Saved scaler:", SCALER_OUT)

df.to_csv(OUT, index=False)
print("Saved final dataset:", OUT)
print("Total features:", len(features))
