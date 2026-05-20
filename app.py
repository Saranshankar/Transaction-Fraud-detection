from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model/fraud_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    amount = float(request.form["amount"])
    time = float(request.form["time"])
    location = float(request.form["location"])
    merchant = float(request.form["merchant"])
    risk = float(request.form["risk"])

    features = [amount, time, location, merchant, risk]

    # fill remaining features to reach 30
    while len(features) < 30:
        features.append(0)

    final = np.array(features).reshape(1, -1)

    prediction = model.predict(final)
    prob = model.predict_proba(final)

    fraud_probability = prob[0][1] * 100

    if prediction[0] == 1:
        result = f"⚠ Fraud Transaction Detected (Probability: {fraud_probability:.2f}%)"
    else:
        result = f"✅ Normal Transaction (Fraud Probability: {fraud_probability:.2f}%)"

    return render_template("index.html", prediction_text=result)


if __name__ == "__main__":
    app.run(debug=False)
