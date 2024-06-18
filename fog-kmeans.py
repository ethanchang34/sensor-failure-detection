import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle

with open('Flagged Data/flagged_data.pkl', 'rb') as file:
    flagged_data = pickle.load(file)

num_sensors = 7 # REPLACE with dynamic code

sensor_ids = flagged_data[next(iter(flagged_data))].keys()

# Initialize an empty list to store flattened data
flattened_data = {sensor_id: [] for sensor_id in sensor_ids}
print(flattened_data)

for day, sensors_data in flagged_data.items():
    for sensor_id, sensor_values in sensors_data.items():
        flattened_data[sensor_id].extend(sensor_values)

# Convert the flattened data into a 2D numpy array
sensor_data = np.array([flattened_data[sensor_id] for sensor_id in sensor_ids])

print(sensor_data)
# print(sensor_data.shape)

def dtw_correlation(data):
    '''Create a correlation matrix between sensors with the value being DTW distance'''
    nsignals = len(data)
    dtw_corrs = np.zeros([nsignals, nsignals])
    for i in range(nsignals):
        for j in range(nsignals):
            distance, _ = fastdtw(data[i], data[j], dist=euclidean)
            dtw_corrs[i, j] = distance

