import pandas as pd
import numpy as np
from scipy import stats
import pickle
import os

with open('Flagged Data/confidence_intervals.pkl', 'rb') as file:
    intervals = pickle.load(file)

with open('Flagged Data/speed_dict_with_anomaly.pkl', 'rb') as file:
    data = pickle.load(file)

def flag_data(data):
    flagged_data = {}
    percent_flagged = {}

    for day, sensor_data in data.items():
        flagged_data[day] = {}
        percent_flagged[day] = {}

        for sensor_id, values in sensor_data.items():
            flagged_data[day][sensor_id] = [None]*len(values)
            count = 0

            for i, value in enumerate(values):
                    lower_bound = intervals[day][sensor_id][i][0]
                    upper_bound = intervals[day][sensor_id][i][1]

                    if value < lower_bound or value > upper_bound:
                        flagged_data[day][sensor_id][i] = 1
                        count += 1
                    else:
                        flagged_data[day][sensor_id][i] = 0

            percent_flagged[day][sensor_id] = count / len(values) # * 100

    # return flagged_data
    return percent_flagged

flagged_data = flag_data(data)

"""Record flagged data"""
with open('Flagged Data/flagged_with_anomaly.pkl', 'wb') as file:
    pickle.dump(flagged_data, file)