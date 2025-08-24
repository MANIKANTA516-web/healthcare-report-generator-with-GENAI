from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from datetime import datetime
import numpy as np
import tensorflow as tf
import requests

app = Flask(__name__)
app.secret_key = "supersecretkey"  # needed for session

# ---------------- ML: TensorFlow Neural Network ---------------- #

tf.random.set_seed(42)
np.random.seed(42)

DISEASES = ["Diabetes", "Hypertension", "Heart Disease", "Flu", "Asthma"]

tf_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(len(DISEASES), activation="sigmoid")
])

# Generate synthetic training data
n_samples = 1000

# Synthetic features: [age_norm, male_flag, female_flag, text_signal]
X_train = np.random.rand(n_samples, 4)

# Synthetic targets: random probabilities for each disease
y_train = np.random.rand(n_samples, len(DISEASES)) * 0.5
# Make some samples have high probability for specific diseases
for i in range(n_samples):
    if i % 5 == 0:
        y_train[i, i % len(DISEASES)] = 0.8 + np.random.rand() * 0.2

# Compile and train the model
tf_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
tf_model.fit(X_train, y_train, epochs=10, verbose=0)

# ---------------- Utilities ---------------- #

def featurize(symptoms: str, medical_history: str, age, gender: str) -> np.ndarray:
    try:
        age = float(age)
    except Exception:
        age = 30.0
    age_norm = max(0.0, min(age / 100.0, 1.0))

    g = (gender or "").strip().lower()
    male_flag = 1.0 if g == "male" else 0.0
    female_flag = 1.0 if g == "female" else 0.0

    total_len = len((symptoms or "").strip()) + len((medical_history or "").strip())
    text_signal = np.log1p(total_len) / 10.0

    return np.array([[age_norm, male_flag, female_flag, text_signal]], dtype=np.float32)


def probs_to_risk(prob: float) -> str:
    if prob > 0.7:
        return "High"
    if prob > 0.4:
        return "Medium"
    return "Low"

# ---------------- Disease Info ---------------- #

DISEASE_INFO = {
    "Diabetes": {
        "medicines": ["Metformin", "Insulin", "Sulfonylureas"],
        "precautions": ["Control sugar intake", "Exercise daily", "Regular checkups"],
    },
    "Hypertension": {
        "medicines": ["ACE inhibitors", "Beta blockers", "Diuretics"],
        "precautions": ["Low salt diet", "Stress management", "Avoid alcohol"],
    },
    "Heart Disease": {
        "medicines": ["Statins", "Aspirin", "Beta blockers"],
        "precautions": ["Balanced diet", "Quit smoking", "Regular exercise"],
    },
    "Flu": {
        "medicines": ["Antiviral drugs", "Paracetamol", "Cough syrups"],
        "precautions": ["Rest well", "Stay hydrated", "Avoid cold drinks"],
    },
    "Asthma": {
        "medicines": ["Inhalers", "Corticosteroids", "Bronchodilators"],
        "precautions": ["Avoid dust", "Carry inhaler", "Breathing exercises"],
    }
}

# ---------------- Gemini API Integration ---------------- #

# Replace with your actual API key from Google AI Studio
GEMINI_API_KEY = "your_actual_api_key_here"

def get_gemini_response(prompt: str) -> str:
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_actual_api_key_here":
        return "AI explanation unavailable. Please add a valid Gemini API key."

    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(apiUrl, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        if 'candidates' in result and result['candidates']:
            content = result['candidates'][0].get('content', {})
            if 'parts' in content and content['parts']:
                return content['parts'][0].get('text', "No response text found.")
        return "Unexpected API response format."
    except requests.exceptions.RequestException as e:
        return f"API Error: {e}"

# ---------------- Report Generator ---------------- #

class ReportGenerator:
    def generate_summary(self, symptoms, medical_history):
        return f"Patient reported symptoms: {symptoms}. Past history: {medical_history or 'No major history reported'}."

    def generate_technical_report(self, predictions):
        return "\n".join([f"{p['disease']} -> Probability: {p['probability']}, Risk: {p['risk_level']}" for p in predictions])

    def generate_lifestyle_recommendations(self, predictions):
        high_risk = [p['disease'] for p in predictions if p['risk_level'] == "High"]
        if high_risk:
            return f"Follow-up with a doctor is recommended for {', '.join(high_risk)}. Maintain healthy diet and exercise."
        return "Maintain a balanced diet, regular exercise, and regular health checkups."

report_generator = ReportGenerator()

# ---------------- ROUTES ---------------- #

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    try:
        if request.method == "GET":
            return jsonify({"message": "Send data using POST method."}), 405

        data = request.get_json()
        symptoms = data.get("symptoms", "")
        medical_history = data.get("medical_history", "")
        age = data.get("age", "")
        gender = data.get("gender", "")

        x = featurize(symptoms, medical_history, age, gender)
        probs = tf_model.predict(x, verbose=0)[0]

        predictions = []
        for disease, p in zip(DISEASES, probs):
            p_round = float(np.round(p, 2))
            info = DISEASE_INFO[disease]
            predictions.append({
                "disease": disease,
                "probability": p_round,
                "risk_level": probs_to_risk(p_round),
                "medicines": info["medicines"],
                "precautions": info["precautions"]
            })

        summary = report_generator.generate_summary(symptoms, medical_history)
        technical_report = report_generator.generate_technical_report(predictions)
        lifestyle_recommendations = report_generator.generate_lifestyle_recommendations(predictions)

        # AI Explanation (Gemini)
        ai_explanation = get_gemini_response(
            f"Provide a simple medical explanation for a patient with symptoms '{symptoms}' "
            f"and risks {', '.join([p['disease']+':'+p['risk_level'] for p in predictions])}."
        )

        # AI Prevention Steps
        prevention_steps = get_gemini_response(
            f"Based on this technical report:\n{technical_report}\n\n"
            "Generate clear, patient-friendly prevention steps and lifestyle changes."
        )

        session["results"] = {
            "patient_data": {
                "symptoms": symptoms,
                "medical_history": medical_history,
                "age": age,
                "gender": gender
            },
            "predictions": predictions,
            "patient_summary": summary,
            "technical_report": technical_report,
            "lifestyle_recommendations": lifestyle_recommendations,
            "ai_explanation": ai_explanation,
            "ai_prevention_steps": prevention_steps,
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        return jsonify({"redirect": url_for("results")})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/results", methods=["GET"])
def results():
    results_data = session.get("results")
    if not results_data:
        return redirect(url_for("index"))
    return render_template("results.html", results=results_data)


if __name__ == "__main__":
    app.run(debug=True)