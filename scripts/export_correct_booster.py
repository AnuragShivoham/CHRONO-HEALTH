import joblib
import json
import xgboost as xgb
import os

print("📥 Loading model from xgb_25k.pkl...")

MODEL_PATH = "Data/processed/xgb_25k.pkl"
OUTPUT_JSON = "Data/processed/xgb_25k_fixed_booster.json"

if not os.path.exists(MODEL_PATH):
    print("❌ ERROR: Model file not found:", MODEL_PATH)
    exit()

model = joblib.load(MODEL_PATH)

print("🔧 Extracting booster...")
booster = model.get_booster()

print("💾 Saving booster in correct JSON format...")
booster.save_model(OUTPUT_JSON)

print("🎉 SUCCESS! Correct booster JSON created:")
print("➡", OUTPUT_JSON)
