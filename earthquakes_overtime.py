import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode
from statsmodels.tsa.stattools import adfuller

df = pd.read_csv("earthquake_data.csv")

required_columns = ['mag', 'depth', 'latitude', 'longitude', 'time']
if not all(col in df.columns for col in required_columns):
    raise ValueError("Dataset missing required columns")

df['time'] = pd.to_datetime(df['time'], errors='coerce')

df = df.dropna(subset=['mag'])

mean_mag = df['mag'].mean()
median_mag = df['mag'].median()
std_mag = df['mag'].std()

mode_result = mode(df['mag'], keepdims=True)
mode_mag = mode_result.mode[0] if mode_result.mode.size > 0 else None

print(f"Mean Magnitude: {mean_mag:.2f}")
print(f"Median Magnitude: {median_mag:.2f}")
print(f"Standard Deviation: {std_mag:.2f}")
print(f"Mode Magnitude: {mode_mag}")

df['year'] = df['time'].dt.year
yearly_counts = df.groupby('year').size()

plt.figure(figsize=(12, 6))
plt.plot(yearly_counts.index, yearly_counts.values, marker='o', linestyle='-')
plt.xlabel("Year")
plt.ylabel("Number of Earthquakes")
plt.title("Earthquake Occurrences Over Time")
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 6))
sns.kdeplot(x=df['longitude'], y=df['latitude'], cmap="Reds", fill=True, alpha=0.7)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Heatmap of Earthquake Occurrences")
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['depth'], y=df['mag'], alpha=0.5)
plt.xlabel("Depth (km)")
plt.ylabel("Magnitude")
plt.title("Earthquake Magnitude vs. Depth")
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))

sns.histplot(df['mag'], bins=50, kde=True, color="blue")
plt.xlabel("Magnitude")
plt.ylabel("Frequency")
plt.title("Earthquake Magnitude Distribution")
plt.grid(True)
plt.show()

adf_result = adfuller(yearly_counts.dropna())

print("ADF Test Statistic:", adf_result[0])
print("p-value:", adf_result[1])
if adf_result[1] < 0.05:
    print("The data is stationary (no trend).")
else:
    print("The data is non-stationary (trend exists).")

correlation_matrix = df[['mag', 'depth']].corr()
print("\nCorrelation Matrix:\n", correlation_matrix)

df.to_csv("processed_earthquake_data.csv", index=False)
print("Processed data saved successfully!")

