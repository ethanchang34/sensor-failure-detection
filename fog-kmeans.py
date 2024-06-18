import numpy as np
from fastdtw import fastdtw
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle

with open('Flagged Data/flagged_data.pkl', 'rb') as file:
    flagged_data = pickle.load(file)

num_sensors = 7 # REPLACE with dynamic code

# Initialize an empty list to store flattened data
flattened_data = {sensor_id: [] for sensor_id in range(num_sensors)}

for day, sensors_data in flagged_data.items():
    for sensor_id, sensor_values in sensors_data.items():
        flattened_data[sensor_id].extend(sensor_values)

# Convert the flattened data into a 2D numpy array
sensor_data = np.array([flattened_data[sensor_id] for sensor_id in range(num_sensors)])

