# export_xgb_onnx.py  -- FINAL XGBOOST → ONNX CONVERSION (onnxmltools)
import os, json, joblib
import pandas as pd
import numpy as np

# ONNXMLTOOLS IMPORT
from onnxmltools.convert.xgboost import convert as convert_xgb
from onnxmltools.convert.common.data_types import FloatTensorType

MODEL_DIR = "Data/processed"
XGB_PATH = f"{MODEL_DIR}/xgb_25k.pkl"
SCALER_PATH = f"{MODEL_DIR}/scaler_25k.pkl"
LABEL_MAP_PATH = f"{MODEL_DIR}/label_mapping_25k.csv"
FEATURE_LIST_PATH = f"{MODEL_DIR}/feature_list_25k.txt"

ONNX_OUT = f"{MODEL_DIR}/xgb_25k.onnx"
SCALER_JSON_OUT = f"{MODEL_DIR}/scaler.json"
LABEL_JSON_OUT = f"{MODEL_DIR}/labels.json"

print("Loading XGBoost model and scaler...")

model = joblib.load(XGB_PATH)
scaler = joblib.load(SCALER_PATH)
labels_df = pd.read_csv(LABEL_MAP_PATH)

with open(FEATURE_LIST_PATH, "r") as f:
    features = [line.strip() for line in f.readlines()]

print("Feature count:", len(features))

# ---- Convert XGBoost → ONNX ----
print("Converting via onnxmltools...")

initial_type = [('input', FloatTensorType([None, len(features)]))]
onnx_model = convert_xgb(model.get_booster(), initial_types=initial_type)

with open(ONNX_OUT, "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Saved ONNX model:", ONNX_OUT)

# ---- Save scaler.json ----
numeric_cols = ["age", "heart_rate", "blood_pressure", "cholesterol_level"]
scaler_json = {
    "mean": scaler.mean_.tolist(),
    "scale": scaler.scale_.tolist(),
    "columns": numeric_cols
}

with open(SCALER_JSON_OUT, "w") as f:
    json.dump(scaler_json, f, indent=2)

print("Saved scaler.json")

# ---- Save labels.json ----
labels_map = {int(row["encoded"]): row["disease"] for _, row in labels_df.iterrows()}

with open(LABEL_JSON_OUT, "w") as f:
    json.dump(labels_map, f, indent=2)

print("Saved labels.json")

print("\n🎉 SUCCESS! ONNX EXPORT COMPLETED (onnxmltools).")
