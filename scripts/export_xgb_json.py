import joblib
import json
import xgboost as xgb

model_path = "Data/processed/xgb_25k_clean.pkl"
output_path = "Data/processed/xgb_25k_model.json"

print("Loading clean model...")
model = joblib.load(model_path)

booster = model.get_booster()

print("Saving booster to JSON...")
booster.save_model(output_path)

print("✔ Saved:", output_path)
