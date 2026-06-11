from flask import Flask, render_template, request
import joblib
from pathlib import Path

app = Flask(__name__)

# app.py가 있는 폴더 기준으로 모델 불러오기
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "final_job_fraud_pipeline.pkl"

# 모델 로드
final_pipeline = joblib.load(MODEL_PATH)


def detect_red_flags(text):
    lower_text = str(text).lower()
    red_flags = []

    suspicious_patterns = {
        "UpfrontPaymentRisk": [
            "training fee",
            "processing fee",
            "registration fee",
            "application fee",
            "deposit",
            "pay upfront",
            "send money",
            "wire transfer",
            "western union"
        ],
        "IdentityTheftRisk": [
            "personal information",
            "bank account",
            "bank details",
            "passport",
            "ssn",
            "social security",
            "driver license",
            "id card",
            "credit card"
        ],
        "UnrealisticOfferRisk": [
            "no experience",
            "quick money",
            "easy money",
            "earn money fast",
            "high income",
            "guaranteed income",
            "weekly payment",
            "earn $",
            "work less"
        ],
        "RemoteScamRisk": [
            "work from home",
            "remote",
            "flexible hours",
            "anywhere",
            "part time from home"
        ],
        "VagueUrgencyRisk": [
            "urgent",
            "urgent hiring",
            "start immediately",
            "immediate start",
            "limited position",
            "no interview",
            "simple work"
        ],
        "CryptoPaymentRisk": [
            "bitcoin",
            "crypto",
            "cryptocurrency"
        ]
    }

    for flag_name, keywords in suspicious_patterns.items():
        if any(keyword in lower_text for keyword in keywords):
            red_flags.append(flag_name)

    if len(str(text).split()) < 30:
        red_flags.append("TooShortDescription")

    if len(red_flags) == 0:
        red_flags.append("NoObviousRuleBasedRedFlag")

    return red_flags


@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        text = request.form.get("job_text", "")

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
            prediction_text = "FraudulentJobPosting"
        else:
            prediction_text = "LegitimateJobPosting"

        red_flags = detect_red_flags(text)

        result = {
            "prediction": prediction_text,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "red_flags": red_flags,
            "input_text": text
        }

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
