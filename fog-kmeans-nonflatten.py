import numpy as np
import torch
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# print(f"Using device: {device}")

with open('Flagged Data/flagged_data.pkl', 'rb') as file:
    flagged_data = pickle.load(file)

sensor_ids = list(flagged_data[next(iter(flagged_data))].keys())

num_sensors = len(sensor_ids)

def dtw_correlation(data):
    '''Create a correlation matrix between sensors with the value being DTW distance'''
    nsignals = len(data)
    dtw_corrs = np.zeros([nsignals, nsignals])
    for i in range(nsignals):
        for j in range(i+1, nsignals):
            print(i,j)
            distance, _ = fastdtw(data[i], data[j])
            dtw_corrs[i, j] = distance
            dtw_corrs[j, i] = distance
    return dtw_corrs

num_clusters = 3

# Initialize an empty list to store cluster counts
cluster_counts = {sensor_id: [] for sensor_id in sensor_ids}
for day, data in flagged_data.items():
    sensor_data = np.array(list(data.values()))
    # Calculate DTW matrix
    dtw_dists = dtw_correlation(sensor_data)
    # Run k-means clustering
    kmeans = KMeans(n_clusters=num_clusters, n_init='auto', random_state=0).fit(dtw_dists)
    # Get the cluster labels
    labels = kmeans.labels_

'''
# Print the cluster centers
print("Cluster Centers:\n", kmeans.cluster_centers_)

# Visualize the clustering result
x_positions = range(len(sensor_ids))
for i in range(num_clusters):
    plt.scatter(x_positions, kmeans.cluster_centers_[i], label=f'Cluster {i}')
plt.xticks(ticks=x_positions, labels=sensor_ids)

plt.legend()
plt.xlabel('Sensor IDs')
plt.ylabel('Cluster Center Value')
plt.title('Cluster Centers Visualization')
plt.show()

# Print the cluster labels for each sensor
for sensor_id, label in zip(sensor_ids, labels):
    print(f'Sensor {sensor_id} is in Cluster {label}')
'''