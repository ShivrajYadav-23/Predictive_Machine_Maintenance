from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os
import json
from datetime import datetime, timedelta
import random

app = Flask(__name__)

# ─── Load Model & Scaler ────────────────────────────────────────────────────
MODEL_PATH  = "model.pkl"
SCALER_PATH = "scaler.pkl"

model  = pickle.load(open(MODEL_PATH,  "rb")) if os.path.exists(MODEL_PATH)  else None
scaler = pickle.load(open(SCALER_PATH, "rb")) if os.path.exists(SCALER_PATH) else None

# ─── In-memory prediction log (replace with DB in production) ───────────────
prediction_log = []

# ─── Model performance metrics (from notebook results) ──────────────────────
MODEL_METRICS = {
    "accuracy": 0.9745,
    "recall":   0.9260,
    "auc_roc":  0.98,
    "precision": 0.7812,
    "f1_score":  0.8447,
    "model_name": "Support Vector Machine (RBF Kernel)",
    "training_samples": 8000,
    "test_samples": 2000,
    "class_balance": {"normal": 96.6, "failure": 3.4}
}

MODEL_COMPARISON = [
    {"model": "Logistic Regression", "accuracy": 0.9650, "recall": 0.7037},
    {"model": "Decision Tree",        "accuracy": 0.9565, "recall": 0.8148},
    {"model": "Random Forest",        "accuracy": 0.9775, "recall": 0.8519},
    {"model": "SVM",                  "accuracy": 0.9745, "recall": 0.9260},
    {"model": "KNN",                  "accuracy": 0.9665, "recall": 0.7593},
]

FEATURE_IMPORTANCE = [
    {"feature": "Torque [Nm]",               "importance": 0.312},
    {"feature": "Rotational speed [rpm]",    "importance": 0.267},
    {"feature": "Tool wear [min]",           "importance": 0.198},
    {"feature": "Process temperature [K]",   "importance": 0.134},
    {"feature": "Air temperature [K]",       "importance": 0.089},
]

# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    try:
        features = np.array([[
            float(data["air_temp"]),
            float(data["process_temp"]),
            float(data["rotational_speed"]),
            float(data["torque"]),
            float(data["tool_wear"]),
        ]])

        if model and scaler:
            scaled   = scaler.transform(features)
            pred     = int(model.predict(scaled)[0])
            prob     = float(model.predict_proba(scaled)[0][1])
        else:
            # Demo mode – deterministic based on inputs
            risk_score = (
                (float(data["tool_wear"])        / 250) * 0.35 +
                (float(data["torque"])           / 70)  * 0.30 +
                (float(data["rotational_speed"]) / 2800)* 0.20 +
                ((float(data["process_temp"]) - 305) / 30) * 0.15
            )
            prob = min(max(risk_score, 0.01), 0.99)
            pred = 1 if prob > 0.5 else 0

        if   prob >= 0.75: risk_level = "CRITICAL"
        elif prob >= 0.50: risk_level = "HIGH"
        elif prob >= 0.25: risk_level = "MODERATE"
        else:              risk_level = "LOW"

        # Maintenance recommendation
        recs = []
        if float(data["tool_wear"]) > 200:
            recs.append("🔧 Immediate tool replacement required (wear > 200 min)")
        if float(data["torque"]) > 60:
            recs.append("⚙️  Reduce operational load – torque exceeds safe threshold")
        if float(data["rotational_speed"]) > 2500:
            recs.append("🔄 Lower rotational speed to reduce mechanical stress")
        if float(data["process_temp"]) > 320:
            recs.append("🌡️  Check cooling system – process temperature elevated")
        if not recs:
            recs.append("✅ Machine operating within normal parameters")

        result = {
            "prediction":  pred,
            "probability": round(prob * 100, 2),
            "risk_level":  risk_level,
            "recommendations": recs,
            "timestamp":   datetime.now().isoformat(),
            "machine_id":  data.get("machine_id", f"MCH-{random.randint(1000,9999)}"),
        }

        prediction_log.append(result)
        if len(prediction_log) > 500:
            prediction_log.pop(0)

        return jsonify({"success": True, "result": result})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/metrics")
def metrics():
    return jsonify({
        "model_metrics":      MODEL_METRICS,
        "model_comparison":   MODEL_COMPARISON,
        "feature_importance": FEATURE_IMPORTANCE,
        "total_predictions":  len(prediction_log),
        "failure_predictions": sum(1 for p in prediction_log if p["prediction"] == 1),
    })


@app.route("/api/history")
def history():
    return jsonify({"history": prediction_log[-50:]})


@app.route("/api/simulate")
def simulate():
    """Generate live sensor telemetry simulation data."""
    now = datetime.now()
    points = []
    for i in range(30):
        t = now - timedelta(minutes=30 - i)
        points.append({
            "time":              t.strftime("%H:%M"),
            "air_temp":          round(298 + random.gauss(0, 1.5), 2),
            "process_temp":      round(308 + random.gauss(0, 2.0), 2),
            "rotational_speed":  round(1500 + random.gauss(0, 80),  1),
            "torque":            round(40   + random.gauss(0, 5),    2),
            "tool_wear":         round(100  + i * 2.5 + random.gauss(0, 3), 1),
            "risk":              round(max(0, 0.05 + i * 0.015 + random.gauss(0, 0.03)), 3),
        })
    return jsonify({"telemetry": points})


@app.route("/api/fleet")
def fleet():
    """Simulated fleet overview for dashboard."""
    machines = []
    statuses = ["NORMAL", "NORMAL", "NORMAL", "MODERATE", "HIGH", "CRITICAL"]
    for i in range(12):
        status = random.choice(statuses)
        machines.append({
            "id":         f"MCH-{1000 + i}",
            "status":     status,
            "tool_wear":  round(random.uniform(20, 240), 1),
            "uptime":     round(random.uniform(85, 99.9), 1),
            "last_check": (datetime.now() - timedelta(minutes=random.randint(1, 120))).strftime("%H:%M"),
        })
    return jsonify({"fleet": machines})


if __name__ == "__main__":
    app.run(debug=True, port=5000)