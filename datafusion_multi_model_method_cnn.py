import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(r"Dataset\range_data_new_32.csv")

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

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model


# Extract input features (X)
#X = df.drop(columns=["target", "Time"]).values
X = df.drop(columns=[ "Time"]).values

# Normalize the input features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Reshape the input features for the CNN model
X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)


if 8 <= df['FullBatteryCapacity'].max() < 10:
    # Load the trained CNN model
    model = load_model(r"CNN_trained_models\trained_rnn_model_20.h5")
elif 10 <= df['FullBatteryCapacity'].max() < 12:
    # Load the trained CNN model
    model = load_model(r"CNN_trained_models\trained_rnn_model_25.h5")
elif 12 <= df['FullBatteryCapacity'].max() < 14:
    # Load the trained CNN model
    model = load_model(r"CNN_trained_models\trained_rnn_model_30.h5")
elif 14 <= df['FullBatteryCapacity'].max() < 16:
    # Load the trained CNN model
    model = load_model(r"CNN_trained_models\trained_rnn_model_35.h5")
elif 16 <= df['FullBatteryCapacity'].max() < 20:
    # Load the trained CNN model
    model = load_model(r"CNN_trained_models\trained_rnn_model_40.h5")

# Make predictions
predictions = model.predict(X_reshaped).flatten()

# Print predictions
print("Predictions:", predictions)


# Calculate the average of the estimated range and predicted output
average_estimated_range_prediction = (estimated_range*0.15 + predictions*0.98)/1

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(time, predictions, label='prediction by CNN')
plt.plot(time, for_representation, label='Estimated remaining range by mathematical model', color='red')
plt.plot(time, average_estimated_range_prediction, label='Fused output of Estimated Range and Predictions', color='green')
plt.plot(time, distance_covered, label='Distance covered', color='blue')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Distance Covered By Vehicle and Average of Estimated Remaining Range and Predictions (vs Time)')
plt.legend()
plt.show()
