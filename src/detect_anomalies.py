import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.distance import geodesic
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import folium
from folium.plugins import MarkerCluster

# ====================
# STEP 1 - LOAD AND CLEANING
# ====================
df = pd.read_csv("data/test_gps_bus.csv")

# Convert time column
df['timestamp'] = pd.to_datetime(df['date_time'])

# Remove null values
df.dropna(inplace=True)

# Remove invalid data
df = df[(df['lat'].between(-90, 90)) & (df['lng'].between(-180, 180))]
df = df[(df['speed'] >= 0) & (df['speed'] <= 120)]
df = df[(df['direction'].between(0, 360))]

# ====================
# STEP 2 - BASIC ANALYSIS
# ====================

# City bounding box (example: Fortaleza)
lat_min, lat_max = -4.0, -3.6
lng_min, lng_max = -38.7, -38.4

bbox_anomalies = df[~((df['lat'].between(lat_min, lat_max)) & (df['lng'].between(lng_min, lng_max)))]

# ====================
# STEP 3 - TEMPORAL AND SPATIAL ANALYSIS
# ====================

# Sort by bus and time
df.sort_values(by=['bus_id', 'timestamp'], inplace=True)

# Calculate distance between consecutive points per bus
def compute_jump_distances(group):
    coords = list(zip(group['lat'], group['lng']))
    distances = [0]
    for i in range(1, len(coords)):
        d = geodesic(coords[i-1], coords[i]).meters
        distances.append(d)
    return pd.Series(distances, index=group.index)

df['jump_distance_m'] = df.groupby('bus_id').apply(compute_jump_distances).reset_index(level=0, drop=True)

# Mark jumps > 1000 meters
df['jump_anomaly'] = df['jump_distance_m'] > 1000

# ====================
# VISUALIZATIONS
# ====================

plt.figure(figsize=(10, 4))
sns.histplot(df['speed'], bins=50, kde=True)
plt.title("Speed Distribution")
plt.xlabel("Speed (km/h)")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='lng', y='lat', hue='jump_anomaly', palette={True: 'red', False: 'blue'}, s=10)
plt.title("Trajectories and Jumps")
plt.grid(True)
plt.tight_layout()
plt.show()

# ====================
# STEP 4 - MACHINE LEARNING: ISOLATION FOREST
# ====================
features = df[['lat', 'lng', 'speed', 'direction', 'jump_distance_m']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

iso = IsolationForest(contamination=0.01, random_state=42)
df['iso_anomaly'] = iso.fit_predict(X_scaled) == -1

# ====================
# STEP 4.2 - DBSCAN FOR SPATIAL CLUSTERING
# ====================
# Using real, non-scaled coordinates
X_geo = df[['lat', 'lng']]
dbscan = DBSCAN(eps=0.01, min_samples=5)  # eps ~1km
df['dbscan_cluster'] = dbscan.fit_predict(X_geo)

# ====================
# ANOMALY SUMMARY
# ====================
print("Jumps > 1000m:", df['jump_anomaly'].sum())
print("Isolation Forest anomalies:", df['iso_anomaly'].sum())
print("Outside bounding box:", bbox_anomalies.shape[0])
print("DBSCAN Clusters:", len(set(df['dbscan_cluster'])) - (1 if -1 in df['dbscan_cluster'].values else 0))

# ====================
# EXPORT ANOMALIES
# ====================
df[df['iso_anomaly'] | df['jump_anomaly']].to_csv("output/anomalies_detected.csv", index=False)

# ====================
# INTERACTIVE MAP WITH FOLIUM
# ====================
mapa = folium.Map(location=[-3.73, -38.54], zoom_start=12)
marker_cluster = MarkerCluster().add_to(mapa)

# Normal points
for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['lng']],
        radius=2,
        color='blue',
        fill=True,
        fill_opacity=0.4
    ).add_to(marker_cluster)

# Anomalous points
anomalias = df[df['iso_anomaly'] | df['jump_anomaly']]

for _, row in anomalias.iterrows():
    folium.Marker(
        location=[row['lat'], row['lng']],
        popup=f"Bus {row['bus_id']}<br>Speed: {row['speed']:.1f} km/h<br>Jump: {row['jump_distance_m']:.1f} m",
        icon=folium.Icon(color='red', icon='exclamation-sign')
    ).add_to(mapa)

mapa.save("output/anomaly_map.html")
print("Map saved at /output as anomaly_map.html")
