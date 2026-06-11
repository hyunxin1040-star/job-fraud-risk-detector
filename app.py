from flask import Flask, render_template, request
import joblib
from pathlib import Path

app = Flask(__name__)

# app.py가 있는 폴더 기준으로 모델 불러오기
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "final_job_fraud_pipeline.pkl"

# 모델 로드
final_pipeline = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        text = request.form["job_text"]

        prediction = final_pipeline.predict([text])[0]
        probability = final_pipeline.predict_proba([text])[0][1]

        risk_score = round(probability * 100, 2)

        if risk_score >= 70:
            risk_level = "HighRisk"
        elif risk_score >= 40:
            risk_level = "MediumRisk"
        else:
            risk_level = "LowRisk"

        if prediction == 1:
            prediction_text = "Fraudulent Job Posting"
        else:
            prediction_text = "Real Job Posting"

        red_flags = []
        lower_text = text.lower()

        suspicious_words = [
            "wire transfer",
            "no experience",
            "quick money",
            "work from home",
            "urgent",
            "personal information",
            "bank account",
            "training fee",
            "processing fee",
            "send money",
            "western union",
            "bitcoin",
            "crypto"
        ]

        for word in suspicious_words:
            if word in lower_text:
                red_flags.append(word)

        if len(red_flags) == 0:
            red_flags.append("No obvious keyword-based red flags detected")

        result = {
            "prediction": prediction_text,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "red_flags": red_flags,
            "input_text": text
        }

    return render_template("index.html", result=result)

app = Flask(__name__)

if __name__ == "__main__":
    app.run()