import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import pickle

with open('Flagged Data/flagged_data.pkl', 'rb') as file:
    flagged_data = pickle.load(file)

data = flagged_data[7] # temporarily work with the first day of data

# Extract sensor IDs and corresponding time series data
sensor_ids = list(data.keys())
time_series_flags = list(data.values())
# print(sensor_ids)

# Compute the DTW distance matrix
num_sensors = len(sensor_ids)
dtw_matrix = np.zeros((num_sensors, num_sensors))

for i in range(num_sensors):
    for j in range(i + 1, num_sensors):
        distance, _ = fastdtw(time_series_flags[i], time_series_flags[j])
        dtw_matrix[i, j] = distance
        dtw_matrix[j, i] = distance

# Convert the DTW matrix to a condensed distance matrix for clustering
condensed_dtw_matrix = squareform(dtw_matrix)

# Perform hierarchical clustering using the condensed DTW distance matrix
Z = linkage(condensed_dtw_matrix, method='ward')

# Plot the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(Z, labels=sensor_ids, leaf_rotation=90)
plt.title('Hierarchical Clustering Dendrogram (DTW)')
plt.xlabel('Sensor ID')
plt.ylabel('Distance')
plt.show()

# Optionally, obtain flat clusters
max_d = 30  # Set this to your desired threshold
clusters = fcluster(Z, max_d, criterion='distance')

# Print the clusters
for sensor_id, cluster_id in zip(sensor_ids, clusters):
    print(f'Sensor ID: {sensor_id}, Cluster ID: {cluster_id}')
