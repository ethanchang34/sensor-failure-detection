import numpy as np
import torch
from fastdtw import fastdtw
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pickle

# Device setup
if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# Load the data
with open('Flagged Data/speed_dict.pkl', 'rb') as file:
    speed_data = pickle.load(file)

sensor_ids = list(speed_data[next(iter(speed_data))].keys())
num_sensors = len(sensor_ids)

# Flatten the data
flattened_data = {sensor_id: [] for sensor_id in sensor_ids}
for day, sensors_data in speed_data.items():
    for sensor_id, sensor_values in sensors_data.items():
        flattened_data[sensor_id].extend(sensor_values)

# Convert flattened data into a 2D numpy array
sensor_data = np.array([flattened_data[sensor_id] for sensor_id in sensor_ids])

# DTW distrance matrix
def dtw_correlation(data):
    '''Create a correlation matrix between sensors using DTW distance'''
    nsignals = len(data)
    dtw_corrs = np.zeros([nsignals, nsignals])
    for i in range(nsignals):
        for j in range(i + 1, nsignals):
            print(i, j)
            distance, _ = fastdtw(data[i], data[j])
            dtw_corrs[i, j] = distance
            dtw_corrs[j, i] = distance
    return dtw_corrs

dtw_dists = dtw_correlation(sensor_data)
print("DTW Distance Matrix:\n", dtw_dists)

# Apply DBSCAN clustering using the precomputed DTW distance matrix
dbscan = DBSCAN(metric="precomputed", eps=80000, min_samples=2)  # You can tune 'eps' and 'min_samples'
labels = dbscan.fit_predict(dtw_dists)

# Print cluster labels
for sensor_id, label in zip(sensor_ids, labels):
    print(f'Sensor {sensor_id} is in Cluster {label}')

# Visualize the clustering result
unique_labels = set(labels)
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

plt.figure(figsize=(10, 6))
for label, color in zip(unique_labels, colors):
    indices = [i for i, lbl in enumerate(labels) if lbl == label]
    plt.scatter(indices, np.zeros(len(indices)), c=[color], label=f'Cluster {label}', s=100)

plt.xticks(ticks=range(len(sensor_ids)), labels=sensor_ids)
plt.legend()
plt.xlabel('Sensor IDs')
plt.ylabel('Cluster')
plt.title('DBSCAN Clustering Result (DTW Distance)')
plt.show()