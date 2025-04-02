import pandas as pd
import sqlite3

# Load the CSV file without headers
file_path = "kopi.csv"  # Replace with your CSV file path
df = pd.read_csv(file_path, header=None)

# Define column names
df.columns = ["Wavelength"] + [f"S{i}" for i in range(1, 14)]

# Reshape the DataFrame: one row per wavelength-intensity pair per sample
df_melted = df.melt(id_vars=["Wavelength"], var_name="S", value_name="Intensity")

# Connect to SQLite database and save the reshaped data
conn = sqlite3.connect("processed_spectra1.db")  # Replace with desired database name
df_melted.to_sql("sample_data", conn, if_exists="replace", index=False)

# Close the connection
conn.close()