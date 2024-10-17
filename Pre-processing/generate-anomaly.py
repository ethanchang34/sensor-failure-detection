import random
import pickle
import numpy as np
import matplotlib.pyplot as plt

anomaly_types = ["complete_failure", "bias", "drift", "accuracy_decline"]

failure_rate = 0.05 # Arbitrary. Find real data to adjust the failure rate

with open('Flagged Data/speed_dict.pkl', 'rb') as file:
    speed_data = pickle.load(file)

anomaly_record = []
labels = {}  # Dictionary to store labels {day: {sensor_id: "anomaly_type"/"normal"}}

def generate_anomalies():
    for day, sensors_data in speed_data.items():
        labels[day] = {}  # Initialize label dictionary for the day
        for sensor_id, sensor_values in sensors_data.items():
            new_data = None
            if random.random() < failure_rate:
                anomaly_type = random.choice(anomaly_types)
                anomaly_record.append((day, sensor_id, anomaly_type))
                labels[day][sensor_id] = anomaly_type
                print(f"Anomaly in sensor {sensor_id} on day {day}: {anomaly_type}")
                
                if anomaly_type == "complete_failure":
                    new_data = generate_complete_failure(sensor_values)
                elif anomaly_type == "bias":
                    new_data = generate_bias(sensor_values)
                elif anomaly_type == "drift":
                    new_data = generate_drift(sensor_values)
                elif anomaly_type == "accuracy_decline":
                    new_data = generate_accuracy_decline(sensor_values)
            else:
                labels[day][sensor_id] = "normal"

            if new_data is not None:
                sensors_data[sensor_id] = new_data
    return speed_data

def generate_complete_failure(day_data):
    '''This function generates a complete failure anomaly for a given day's sensor data.
    It replaces all sensor values with a constant extreme value'''
    new_data = day_data.copy()
    extreme_value = 0
    if random.random() < 0.5:
        extreme_value = max(day_data)+random.random()*5+10 # Some value between 10 and 15 above the maximum speed during the day
    for i in range(len(new_data)):
        new_data[i] = extreme_value
    return new_data

def generate_bias(day_data):
    '''This function generates a bias anomaly for a given day's sensor data.
    It adds or subtracts some constant value from the data.'''
    new_data = day_data.copy()
    bias = max(day_data)*(0.1*random.random()+0.2)
    if random.random() < 0.5:
        bias *= -1
    for i in range(len(new_data)):
        new_data[i] += bias
    return new_data

def generate_drift(day_data):
    '''This function generates a drift anomaly for a given day's sensor data.
    It adds or subtracts an increasing value from the data.'''
    new_data = day_data.copy()
    n = len(new_data)
    drift_increment = max(day_data)/(2*n)
    if random.random() < 0.5:
        drift_increment *= -1
    for i in range(len(new_data)):
        new_data[i] += i * drift_increment
    return new_data

def generate_accuracy_decline(day_data):
    '''This function generates an accuracy decline anomaly for a given day's sensor data.
    It adds increasing variance or error to the data.'''
    new_data = day_data.copy()
    for i in range(len(new_data)):
        variance = 0.1*i #CHECK THIS VALUE
        noise = np.random.normal(0, variance)
        new_data[i] += noise
    return new_data

speed_dict_with_anomaly = generate_anomalies()

print("num anomalies:", len(anomaly_record))

with open('Flagged Data/speed_dict_with_anomaly.pkl', 'wb') as file:
    pickle.dump(speed_dict_with_anomaly, file)

with open('Flagged Data/labels.pkl', 'wb') as file:
    pickle.dump(labels, file)

# print(labels[7])

# for day, sensor_id, anomaly_type in anomaly_record:
#     sensor_data = speed_dict_with_anomaly[day][sensor_id]
#     plt.plot(range(len(sensor_data)), sensor_data, label=f"Sensor {sensor_id}")
#     plt.title({anomaly_type})
#     plt.show()