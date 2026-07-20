from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)
import sys
import sklearn

print("Python Executable:", sys.executable)
print("Python Version:", sys.version)
print("Scikit-learn:", sklearn.__version__)
# Load trained model
model = joblib.load("car_price_prediction.pkl")
feature_names = [
    "Present Price",
    "KM Driven",
    "Owner",
    "Car Age",
    "Fuel (Diesel)",
    "Fuel (Petrol)",
    "Seller Type",
    "Transmission"
]

feature_importance = []

if hasattr(model, "feature_importances_"):

    values = model.feature_importances_

    total = values.sum()

    for name, val in zip(feature_names, values):

        feature_importance.append({

            "name": name,

            "percent": round((val/total)*100,2)

        })

feature_importance = sorted(
    feature_importance,
    key=lambda x:x["percent"],
    reverse=True
)
print(type(model))
print("Feature Importance:")
for feature in feature_importance:
    print(f"  {feature['name']}: {feature['percent']}%")

# Model Performance Metrics
r2 = 0.9694
mae = 0.5607
rmse = 0.8394
cars = 301

@app.route("/")
def home():
    return render_template(
        "index.html",
        feature_importance=feature_importance,
        r2=r2,
        mae=mae,
        rmse=rmse,
        cars=cars
    )


@app.route("/predict", methods=["POST"])
def predict():

    present_price = float(request.form["present_price"])
    kms_driven = float(request.form["kms_driven"])
    owner = int(request.form["owner"])
    car_age = int(request.form["car_age"])

    fuel = request.form["fuel"]
    seller = request.form["seller"]
    transmission = request.form["transmission"]

    # One-hot encoding (must match training data)
    fuel_diesel = 0
    fuel_petrol = 0

    if fuel == "Diesel":
        fuel_diesel = 1
    elif fuel == "Petrol":
        fuel_petrol = 1
    # CNG -> both remain 0

    seller_individual = 0
    if seller == "Individual":
        seller_individual = 1
    # Dealer -> 0

    transmission_manual = 0
    if transmission == "Manual":
        transmission_manual = 1
    # Automatic -> 0

    features = np.array([[
        present_price,
        kms_driven,
        owner,
        car_age,
        fuel_diesel,
        fuel_petrol,
        seller_individual,
        transmission_manual
    ]])

    prediction = model.predict(features)[0]
    prediction = round(prediction, 2)

    return render_template(
    "index.html",

    prediction_text=f"Estimated Selling Price: ₹ {prediction} Lakhs",

    present_price=present_price,
    kms_driven=kms_driven,
    owner=owner,
    car_age=car_age,
    fuel=fuel,
    seller=seller,
    transmission=transmission,

    feature_importance=feature_importance,

    r2=r2,
    mae=mae,
    rmse=rmse,
    cars=cars
)


if __name__ == "__main__":
    app.run(debug=True)