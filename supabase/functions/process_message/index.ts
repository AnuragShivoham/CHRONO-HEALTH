// FIXED: Generate 46 features exactly matching training model
const FEATURE_NAMES = [
  "age",
  "gender",
  "smoker",
  "heart_rate",
  "blood_pressure",
  "cholesterol_level",
  "fever",
  "cough",
  "fatigue",
  "shortness_of_breath",
  "headache",
  "runny_nose",
  "sore_throat",
  "chest_pain",
  "body_ache",
  "nausea",
  "vomiting",
  "diarrhea",
  "dizziness",
  "chills",
  "loss_of_smell",
  "loss_of_taste",
  "wheezing",
  "rash",
  "eye_irritation",
  "ear_pain",
  "sweating",
  "joint_pain",
  "abdominal_pain",
  "back_pain",
  "blurred_vision",
  "dry_cough",
  "wet_cough",
  "sinus_pressure",
  "sneezing",
  "rapid_heartbeat",
  "slow_heartbeat",
  "dehydration",
  "loss_of_appetite",
  "sleep_disturbance",
  "anxiety",
  "irritability",
  "muscle_spasm",
  "skin_redness",
  "itchiness",
  "breathing_difficulty"
];

// MAP symptoms to severity (0–3)
const severityMap = {
  "severe": 3,
  "high": 3,
  "bad": 3,
  "moderate": 2,
  "medium": 2,
  "mild": 1,
  "slight": 1,
};

function extractSeverity(msg: string) {
  msg = msg.toLowerCase();
  for (const [key, val] of Object.entries(severityMap)) {
    if (msg.includes(key)) return val;
  }
  return 1; // default mild
}

function buildFeatureVector(message: string) {
  const vec = Array(FEATURE_NAMES.length).fill(0);
  const msg = message.toLowerCase();

  // Default values for vitals (can adjust later)
  vec[0] = 25;      // age
  vec[1] = 1;       // gender (1=unknown)
  vec[2] = 0;       // smoker
  vec[3] = 80;      // heart_rate
  vec[4] = 120;     // blood_pressure
  vec[5] = 180;     // cholesterol_level

  const severity = extractSeverity(msg);

  FEATURE_NAMES.forEach((name, i) => {
    if (i < 6) return; // vitals already set

    const symptom = name.replace(/_/g, " ");

    if (msg.includes(symptom)) {
      vec[i] = severity;   // set symptom severity
    }
  });

  return vec;
}
