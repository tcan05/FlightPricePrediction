import os
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


# ------------------ FUNCTIONS -------------------
def evaluate(prediction, y_true, num_features):

    mae = mean_absolute_error(y_true, prediction)
    mse = mean_squared_error(y_true, prediction)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, prediction)

    num_samples = len(y_true)
    adj_r2 = 1 - (1 - r2) * (num_samples - 1) / (num_samples - num_features - 1)

    print(f"Mean Absolute Error    : {mae:.3f}")
    print(f"Mean Squared Error     : {mse:.3f}")
    print(f"Root Mean Squared Error: {rmse:.3f}")
    print(f"R^2 Score              : {r2:.6f}")
    print(f"Adjusted R^2 Score     : {adj_r2:.6f}")


def plot_predictions(y_true, y_pred, model_name):

    plt.figure(figsize = (8,8))
    plt.scatter(y_true, y_pred, alpha = 0.5, color = "red", label = "Predicted Values")

    plt.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        color = "blue",
        label = "Perfect Prediction Line"
    )

    plt.xlabel("Real Prices")
    plt.ylabel("Predicted Prices")
    plt.title(f"Real vs Predicted Prices ({model_name})")

    plt.legend()
    plt.show()
# ---------------------------------------------


start = time.time()

script_dir = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(script_dir, "flight_dataset_cleaned.csv")

dataset = pd.read_csv(filepath)

# Display dataset information
print(dataset.shape)
print(dataset.isnull().sum())
print(dataset.dtypes)

# Dataset cleaning
dataset = dataset.drop(columns = ["flight"], axis = 1)

# Seperating the target feature and one-hot encoding the categorical features
X = dataset.drop(columns = ["price"], axis = 1)
X = pd.get_dummies(X, drop_first = True)
y = dataset["price"]

# Dataset splitting and model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

model_linear = LinearRegression()
model_forest = RandomForestRegressor()
model_xgb = XGBRegressor()

model_linear.fit(X_train, y_train)
model_forest.fit(X_train, y_train)
model_xgb.fit(X_train, y_train)

pred_linear = model_linear.predict(X_test)
pred_forest = model_forest.predict(X_test)
pred_xgb = model_xgb.predict(X_test)

num_features = X_train.shape[1]
end = time.time()

# Evaluation and visualization
print("Linear Regression Evaluation")
evaluate(pred_linear, y_test, num_features)

print("\nRandom Forest Evaluation")
evaluate(pred_forest, y_test, num_features)

print("\nXGBoost Evaluation")
evaluate(pred_xgb, y_test, num_features)

print(f"\nExecution Time: {end - start:.2f} seconds")

plot_predictions(y_test, pred_linear, "Linear Regression")
plot_predictions(y_test, pred_forest, "Random Forest")
plot_predictions(y_test, pred_xgb, "XGBoost")