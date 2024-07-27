import pickle
import numpy as np

with open('Flagged Data/flagged_data.pkl', 'rb') as file:
    flagged_data = pickle.load(file)

sensor_ids = list(flagged_data[next(iter(flagged_data))].keys())

num_sensors = len(sensor_ids)

for day, data in flagged_data.items():
    sensor_data = np.array(list(data.values()))
    print(sensor_data)
    print(sensor_data.shape)