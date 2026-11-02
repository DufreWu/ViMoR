import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load and Prepare
df = pd.read_csv('latency_fitting_data_2.csv')
X = np.column_stack((
    df['l_p'],
    df['l_d'],
    df['cpu_freq_hz'],
    df['gpu_freq_hz']
))
y = df['e2e_latency']

# 2. Split data: Train on 80%, Test on 20%
# This proves the model works on data it has never seen before!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Validate
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Model Accuracy (R²): {r2:.4f}")
print(f"MSE: {mse:.6f} seconds^2")

# 5. Save your coefficients for later use
import joblib
joblib.dump(model, 'latency_predictor.pkl')
print("Model saved as latency_predictor.pkl")