from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# LOAD MODEL + SCALER

model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

# Features EXACTLY as training (without 'step')
FEATURE_COLUMNS = [
    'amount','oldbalanceOrg','newbalanceOrig',
    'oldbalanceDest','newbalanceDest',
    'isFlaggedFraud','errorOrig','errorDest',
    'type_CASH_OUT','type_DEBIT','type_PAYMENT','type_TRANSFER'
]

# HOME PAGE
@app.route("/")
def home():
    return render_template("index.html")

# PREDICT FUNCTION
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # -------- GET FORM VALUES --------
        amount = float(request.form.get("amount", 0))
        oldbalanceOrg = float(request.form.get("oldbalanceOrg", 0))
        newbalanceOrig = float(request.form.get("newbalanceOrig", 0))
        oldbalanceDest = float(request.form.get("oldbalanceDest", 0))
        newbalanceDest = float(request.form.get("newbalanceDest", 0))
        type_value = request.form.get("type")

        # -------- FEATURE ENGINEERING --------
        errorOrig = oldbalanceOrg - newbalanceOrig - amount
        errorDest = newbalanceDest - oldbalanceDest - amount

        # -------- TYPE ENCODING --------
        type_CASH_OUT = 1 if type_value == "CASH_OUT" else 0
        type_DEBIT = 1 if type_value == "DEBIT" else 0
        type_PAYMENT = 1 if type_value == "PAYMENT" else 0
        type_TRANSFER = 1 if type_value == "TRANSFER" else 0

        # Default value for flag
        isFlaggedFraud = 0

        # -------- CREATE DATAFRAME --------
        input_data = [
            amount, oldbalanceOrg, newbalanceOrig,
            oldbalanceDest, newbalanceDest,
            isFlaggedFraud, errorOrig, errorDest,
            type_CASH_OUT, type_DEBIT, type_PAYMENT, type_TRANSFER
        ]

        input_df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)

        # -------- SCALE INPUT --------
        scaled_data = scaler.transform(input_df)

        # -------- PREDICT --------
        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0][1]

        result = "Fraud Transaction ❌" if prediction == 1 else "Legitimate Transaction ✅"

        return render_template(
            "result.html",
            prediction=result,
            probability=round(probability*100,2)
        )

    except Exception as e:
        return render_template(
            "result.html",
            prediction="Error occurred",
            probability=str(e)
        )

# RUN APP
if __name__ == "__main__":
    app.run(debug=True)