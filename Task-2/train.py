import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
y_pred = gb_model.predict(X_test)
plt.figure(figsize=(7,7))

plt.scatter(y_test, y_pred, alpha=0.7)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--',
    linewidth=2
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")

plt.grid(True)
plt.savefig("static/actual_vs_predicted.png")
plt.close()