import joblib
import m2cgen as m2c
import os
import sys

MODEL_DIR = "Data/processed"
#MODEL_PATH = os.path.join(MODEL_DIR, "xgb_25k.pkl")
MODEL_PATH = "Data/processed/xgb_25k_clean.pkl"
JS_OUT = os.path.join(MODEL_DIR, "xgb_model.js")

print("🔍 Checking model path:", MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    print("❌ ERROR: Model file not found!")
    sys.exit(1)

print("📥 Loading XGBoost model...")
model = joblib.load(MODEL_PATH)

# ------------------------------------------------------------------
# 🔥 IMPORTANT FIX — Remove legacy attributes XGBoost no longer supports
# ------------------------------------------------------------------
if hasattr(model, "use_label_encoder"):
    print("⚠️ Removing deprecated parameter: use_label_encoder")
    delattr(model, "use_label_encoder")

if hasattr(model, "_le"):
    print("⚠️ Removing deprecated internal label encoder")
    delattr(model, "_le")

# ------------------------------------------------------------------
# Convert model → JS
# ------------------------------------------------------------------
print("🔄 Converting XGBoost model → JavaScript using m2cgen...")
try:
    js_code = m2c.export_to_javascript(model)
except Exception as e:
    print("❌ Conversion failed:")
    print(e)
    sys.exit(1)

print("💾 Saving JS model →", JS_OUT)
with open(JS_OUT, "w", encoding="utf-8") as f:
    f.write(js_code)

print("\n🎉 SUCCESS! xgb_model.js created.")
