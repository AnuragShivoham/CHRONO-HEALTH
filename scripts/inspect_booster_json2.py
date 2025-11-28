import json
fpath = "Data/processed/xgb_25k_fixed_booster.json"  # adjust to your actual file name
with open(fpath, "r", encoding="utf-8") as f:
    data = json.load(f)
print("Top‐level keys:", list(data.keys()))
if "learner" not in data:
    print("No 'learner' key found.")
    # print small subset
print("Printing nested keys and sample node:")
import itertools
for k,v in data.items():
    print(k, type(v))
    if isinstance(v, dict):
        print("  Subkeys:", list(v.keys())[:5])
    break
