import json

with open("Data/processed/xgb_25k_model.json", "r") as f:
    data = json.load(f)

print("Top-level keys:")
print(list(data.keys()))

print("\nExample content:")
print(str(data)[:2000])
