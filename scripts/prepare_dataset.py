import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

IN = "Data/interim/cleaned_25k.csv"
OUT = "Data/interim/prepared_25k.csv"
LABEL_OUT = "Data/processed/label_mapping_25k.csv"

os.makedirs("Data/interim", exist_ok=True)
os.makedirs("Data/processed", exist_ok=True)

df = pd.read_csv(IN)

# Encode gender to 0/1
df["gender"] = df["gender"].astype(str).str.lower().map(
    lambda x: 0 if x in ["male","m","0"] else 1
)

# Numeric vitals
num_cols = ["age","smoker","heart_rate","blood_pressure","cholesterol_level"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(df[c].median()).astype(int)

# Symptoms → int
symptoms = [c for c in df.columns if c not in num_cols + ["gender","disease"]]
for c in symptoms:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

# Encode disease
le = LabelEncoder()
df["disease_encoded"] = le.fit_transform(df["disease"])

# Save mapping
map_df = pd.DataFrame({"disease": le.classes_, "encoded": range(len(le.classes_))})
map_df.to_csv(LABEL_OUT, index=False)
print("Saved label mapping →", LABEL_OUT)

df.to_csv(OUT, index=False)
print("Saved prepared file →", OUT)
