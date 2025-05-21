from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route("/")
def index():
    return render_template("form.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        fields = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ]
        data = [float(request.form.get(field)) for field in fields]
        prediction = model.predict([data])[0]
        probability = model.predict_proba([data])[0][1]  # prawdopodobieństwo choroby (klasa 1)

        # Interpretacja
        risk_level = ""
        if probability >= 0.8:
            risk_level = "Bardzo wysokie ryzyko cukrzycy"
        elif probability >= 0.5:
            risk_level = "Umiarkowane ryzyko cukrzycy"
        elif probability >= 0.2:
            risk_level = "Niskie ryzyko cukrzycy"
        else:
            risk_level = "Bardzo niskie ryzyko cukrzycy"

        detailed_result = f"{risk_level} ({probability:.2%} prawdopodobieństwa)"
        return render_template("form.html", prediction=detailed_result)

    except Exception as e:
        return render_template("form.html", prediction=f"Błąd danych: {e}")

if __name__ == "__main__":
    app.run(debug=True)
