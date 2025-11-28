import pandas as pd
import os

RAW = "Data/raw/synthetic_dataset_25k_40symptoms.csv"
OUT = "Data/interim/cleaned_25k.csv"

os.makedirs("Data/interim", exist_ok=True)

print("Loading raw CSV...")
df = pd.read_csv(RAW)

# Standardize column names
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

# Remove accidental chronic-disease leakage (rare, safety measure)
DROP = ["diabetes", "hypertension", "cancer"]
for col in DROP:
    if col in df.columns:
        df = df.drop(columns=[col])
        print(f"Dropped: {col}")

# Verify target
if "disease" not in df.columns:
    raise SystemExit("ERROR: 'disease' column missing.")

df.to_csv(OUT, index=False)
print("Saved cleaned file →", OUT)
