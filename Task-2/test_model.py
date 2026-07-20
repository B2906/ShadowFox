import sklearn
import joblib

print("Python version:")
import sys
print(sys.version)

print("Scikit-learn:", sklearn.__version__)

model = joblib.load("car_price_prediction.pkl")

print(type(model))
print("Model loaded successfully!")