import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(r"Dataset\range_data_new_20.csv")

# Mathematical model for range estimation.
time = df['Time']  # time in seconds
fuel_efficiency = df['UsedCapacity'] * 1000 / df['DistanceKm']
estimated_range = df['PT.BattHV.Energy'] * 1000 / fuel_efficiency  # the estimated remaining range of vehicle from current situation
distance_covered = df['DistanceKm']  # distance covered by the vehicle at present
for_representation = distance_covered + estimated_range

'''
plt.plot(time, for_representation, label='Estimated remaining range by mathematical model', color='green')
plt.plot(time, distance_covered, label='Distance covered', color='blue')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Distance covered By Vehicle and Estimated Remaining Range of the Vehicle (vs Time)')
plt.legend()
plt.show()
'''

# Extract input features (X)
X = df.drop(columns=["target", "Time"]).values

# Extract true target values (y_true)
y_true = df["target"].values

# Normalize the input features
scaler = MinMaxScaler()
# X_scaled = scaler.fit_transform(X)

# Reshape the input features for the RNN model
X_reshaped = X.reshape(X.shape[0], 1, X.shape[1])

# Load the trained RNN model
model = load_model("trained_cnn_model.h5")

# Make predictions
predictions = model.predict(X_reshaped).flatten()

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_true, predictions))
print("Root Mean Squared Error (RMSE):", rmse)

# Calculate the average of the estimated range and predicted output
average_estimated_range_prediction = (estimated_range*1 + predictions*4) / 5

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(time, predictions, label='prediction by RNN')
plt.plot(time, for_representation, label='Estimated remaining range by mathematical model', color='red')
plt.plot(time, average_estimated_range_prediction, label='Fused output of Estimated Range and Predictions', color='green')
plt.plot(time, distance_covered, label='Distance covered', color='blue')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Distance Covered By Vehicle and Average of Estimated Remaining Range and Predictions (vs Time)')
plt.legend()
plt.show()
