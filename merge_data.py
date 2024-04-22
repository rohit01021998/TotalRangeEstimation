import pandas as pd

# Read the data from the CSV file
df = pd.read_csv(r"Dataset\range_data_new_25.csv")

# Find the max value of DistanceKm column
max_distance = df["DistanceKm"].max()

# Create a new column called "target" and fill it with the max value of DistanceKm
df["target"] = max_distance

# Save the modified DataFrame back to the same CSV file
df.to_csv(r"Dataset\range_data_new_25.csv", index=False)

print("Data saved back to the same CSV file.")

#-----------------------------------------------------------------------------------
# Read the data from the CSV file
df = pd.read_csv(r"Dataset\range_data_new_35.csv")

# Find the max value of DistanceKm column
max_distance = df["DistanceKm"].max()

# Create a new column called "target" and fill it with the max value of DistanceKm
df["target"] = max_distance

# Save the modified DataFrame back to the same CSV file
df.to_csv(r"Dataset\range_data_new_35.csv", index=False)

print("Data saved back to the same CSV file.")



#--------------------------------------------------------------------------------------------

# File paths for the CSV files
file_paths = [r"Dataset\dataset.csv", r"Dataset\range_data_new_35.csv", r"Dataset\range_data_new_25.csv"]

# Read CSV files into pandas DataFrames
dfs = [pd.read_csv(file) for file in file_paths]

# Merge DataFrames using pandas.concat()
merged_df = pd.concat(dfs, ignore_index=True)

# Write the merged DataFrame to a new CSV file
merged_df.to_csv("Dataset\dataset.csv", index=False)

print("Merged CSV file saved successfully.")
