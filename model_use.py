import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(r"Dataset\dataset.csv")

# Extract input features (X)
X = df.drop(columns=["target","Time"]).values

y_true = df["target"].values

# Normalize the input features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Reshape the input features for the RNN model
X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

# Load the trained RNN model
model = load_model("trained_rnn_model.h5")

# Make predictions
predictions = model.predict(X_reshaped)

# Print predictions
print("Predictions:")
print(predictions)


# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_true, predictions))

print("Root Mean Squared Error (RMSE):", rmse)

# plotting

plt.figure(figsize=(10, 6))
plt.plot(y_true, label='True Values', color='blue')
plt.plot(predictions, label='Predictions', color='red')
plt.xlabel('Time')
plt.ylabel('Target Value')
plt.title('True Values vs. Predictions')
plt.legend()
plt.grid(True)
plt.show()