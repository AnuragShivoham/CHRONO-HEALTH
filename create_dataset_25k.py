import pandas as pd
import numpy as np
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------

NUM_ROWS = 25000

diseases = [
    "influenza", "common_cold", "pneumonia", "asthma",
    "bronchitis", "covid19", "allergy", "tuberculosis"
]

# 40 medically meaningful symptoms
symptoms = [
    "fever", "cough", "fatigue", "shortness_of_breath", "headache",
    "runny_nose", "sore_throat", "chest_pain", "body_ache",
    "nausea", "vomiting", "diarrhea", "dizziness", "chills",
    "loss_of_smell", "loss_of_taste", "wheezing", "rash",
    "eye_irritation", "ear_pain", "sweating", "joint_pain",
    "abdominal_pain", "back_pain", "blurred_vision", "dry_cough",
    "wet_cough", "sinus_pressure", "sneezing", "rapid_heartbeat",
    "slow_heartbeat", "dehydration", "loss_of_appetite",
    "sleep_disturbance", "anxiety", "irritability",
    "muscle_spasm", "skin_redness", "itchiness",
    "breathing_difficulty"
]

# -----------------------------
# Helper: Assign disease logically
# -----------------------------

def assign_disease(row):
    # 🔵 Influenza
    if row["fever"] >= 2 and row["cough"] >= 2 and row["body_ache"] >= 2:
        return "influenza"

    # 🟢 Common Cold
    if row["runny_nose"] >= 2 and row["sore_throat"] >= 1:
        return "common_cold"

    # 🔴 Pneumonia
    if (row["shortness_of_breath"] >= 2 or row["breathing_difficulty"] >= 2) and row["chest_pain"] >= 1:
        return "pneumonia"

    # 🟠 Asthma
    if row["wheezing"] >= 2 or row["dry_cough"] >= 2:
        return "asthma"

    # 🟤 Bronchitis
    if row["wet_cough"] >= 2 and row["chest_pain"] >= 2:
        return "bronchitis"

    # 🟡 COVID-19
    if row["fever"] >= 2 and row["loss_of_smell"] >= 1:
        return "covid19"

    # 💚 Allergy
    if row["rash"] >= 2 or row["itchiness"] >= 2 or row["eye_irritation"] >= 2:
        return "allergy"

    # ⚫ Tuberculosis
    if row["cough"] >= 2 and row["fatigue"] >= 2:
        return "tuberculosis"

    return np.random.choice(diseases)

# -----------------------------
# Generate Data
# -----------------------------

rows = []
print("Generating dataset...")

for _ in tqdm(range(NUM_ROWS)):
    person = {
        "age": np.random.randint(5, 85),
        "gender": np.random.choice(["male", "female"]),
        "smoker": np.random.choice([0, 1]),
        "diabetes": np.random.choice([0, 1]),
        "heart_rate": np.random.randint(60, 140),
        "blood_pressure": np.random.randint(90, 180),
        "cholesterol_level": np.random.randint(120, 300),
    }

    # Add all symptoms (0–3 severity)
    for s in symptoms:
        person[s] = np.random.randint(0, 4)

    # Assign disease outcome
    person["disease"] = assign_disease(person)

    rows.append(person)

df = pd.DataFrame(rows)

# -----------------------------
# Save to CSV
# -----------------------------

output_path = "synthetic_dataset_25k_40symptoms.csv"
df.to_csv(output_path, index=False)

print("\n🎉 Dataset generated successfully!")
print(f"📁 Saved to: {output_path}")
print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")
