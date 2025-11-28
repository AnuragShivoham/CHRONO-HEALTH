import joblib
import xgboost as xgb
import pandas as pd

print("🔍 Loading original XGB model...")
old_model = joblib.load("Data/processed/xgb_25k.pkl")
old_booster = old_model.get_booster()

print("📥 Loading label mapping...")
label_df = pd.read_csv("Data/processed/label_mapping_25k.csv")
classes = label_df["disease"].tolist()
num_classes = len(classes)

print("Detected classes:", classes)

print("🛠 Extracting booster config...")
config_json = old_booster.save_config()

print("🔄 Creating clean booster...")
clean_booster = xgb.Booster()
clean_booster.load_config(config_json)

num_features = old_model.n_features_in_
print("Detected feature count:", num_features)

# ************* CRITICAL FIX (MULTICLASS REQUIRED) *************
clean_booster.set_param("num_feature", num_features)
clean_booster.set_param("num_class", num_classes)
clean_booster.set_param("objective", "multi:softprob")
# **************************************************************

print("🧱 Building clean XGBClassifier shell...")

clean_model = xgb.XGBClassifier()
clean_model._Booster = clean_booster

# Required metadata
clean_model.classes_ = classes
clean_model.n_classes_ = num_classes
clean_model._n_classes = num_classes

clean_model._n_features_in = num_features
clean_model._feature_names = [f"f{i}" for i in range(num_features)]

# Disable label encoder completely
clean_model._le = None

out_path = "Data/processed/xgb_25k_clean.pkl"
joblib.dump(clean_model, out_path)

print("\n🎉 CLEAN MODEL SAVED SUCCESSFULLY!")
print("➡", out_path)
