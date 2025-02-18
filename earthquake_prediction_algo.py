import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta

df = pd.read_csv("earthquake_data.csv")

df['time'] = pd.to_datetime(df['time'], errors='coerce')

df = df.dropna(subset=['time'])

df = df.drop_duplicates(subset=['time'])

df.set_index('time', inplace=True)

numeric_columns = df.select_dtypes(include=[np.number]).columns
df = df[numeric_columns]

df = df.resample('D').mean()

df.fillna(method='ffill', inplace=True)

# ........................................... ARIMA Model for Time Series Forecasting
order = (5, 1, 0)  #......................... Adjust ARIMA parameters as needed
model = ARIMA(df['mag'], order=order)
model_fit = model.fit()

future_days = 30
future_dates = [df.index[-1] + timedelta(days=i) for i in range(1, future_days + 1)]
forecast = model_fit.forecast(steps=future_days)

future_df = pd.DataFrame({'time': future_dates, 'predicted_mag': forecast})
future_df.set_index('time', inplace=True)

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['mag'], label="Past Magnitude", color="blue")
plt.plot(future_df.index, future_df['predicted_mag'], label="Predicted Magnitude", color="red", linestyle="dashed")
plt.xlabel("Date")
plt.ylabel("Magnitude")
plt.title("Earthquake Magnitude Prediction")
plt.legend()
plt.show()


map_center = [df['latitude'].mean(), df['longitude'].mean()]
earthquake_map = folium.Map(location=map_center, zoom_start=3, tiles="Esri WorldImagery")


for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=row['mag'],
        color='blue',
        fill=True,
        fill_color='blue',
        fill_opacity=0.5,
    ).add_to(earthquake_map)

for _, row in future_df.iterrows():
    sample_location = df.sample(1) 
    folium.CircleMarker(
        location=[sample_location['latitude'].values[0], sample_location['longitude'].values[0]],
        radius=row['predicted_mag'],
        color='red',
        fill=True,
        fill_color='red',
        fill_opacity=0.5,
    ).add_to(earthquake_map)


earthquake_map.save("earthquake_prediction_map.html")

print("Prediction complete. Open 'earthquake_prediction_map.html' to view the map.")
