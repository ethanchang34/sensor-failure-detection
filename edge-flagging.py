import pandas as pd
import numpy as np
from scipy import stats
import pickle
import os


def get_speed_from_csv(directory):
    '''Read all csv data in directory and return a tuple with the sensor id and its speed data '''
    speed_table = [] # Stores each sensor's id and speed matrix (device_id, Dataframe of speeds)
    min_num_days = float('inf')
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)
            # Convert 'index' column to datetime
            df['index'] = pd.to_datetime(df['index'])
            # Extract date and time components into separate columns
            df['date'] = df['index'].dt.date
            df['time'] = df['index'].dt.time
            # Pivot the DataFrame
            speed_matrix = df.pivot(index='date', columns='time', values='speed') # Timepoints are columns and each day is a row
            if len(speed_matrix) < min_num_days:
                min_num_days = len(speed_matrix)
            # Fill any missing values with 0 (if needed)
            speed_matrix.fillna(0, inplace=True)
            speed_table.append((df['device_id'][0], speed_matrix))
    # Clean up data by ensuring same size (number of days)
    speed_table = truncate_data(speed_table, min_num_days)
    return speed_table

def truncate_data(speed_table, min_num_days):
    '''Truncate sensor data to ensure the same number of days in data based on the sensor with the least data'''
    for i, (device_id, df) in enumerate(speed_table):
        speed_table[i] = (device_id, df[:min_num_days])
    return speed_table

def calc_conf_interval(conf_interval_dict, speed_table):
    '''Every day, calculate the confidence interval based on past 7 days of data for each sensor'''
    # Iterate through each day, i (days are 0-indexed)
    num_days = speed_table[0][1].shape[0]
    for day in range(7, num_days+1):
        conf_interval_dict[day] = {}
        for device_id, speed_matrix in speed_table: # Do for each sensor
            # Get last 7 days of data
            speed_matrix = speed_matrix.iloc[day-7:day, :]
            # Calculate the mean and standard deviation across each column
            mean_values = speed_matrix.mean(axis=0)
            std_dev_values = speed_matrix.std(axis=0)
            # Calculate the number of samples (days) for each column
            n_samples = speed_matrix.shape[0]
            # Set the confidence level
            confidence_level = 0.95
            # Calculate the margin of error
            margin_of_error = std_dev_values * stats.t.ppf((1 + confidence_level) / 2, n_samples - 1) / np.sqrt(n_samples)
            # Calculate the confidence interval lower and upper bounds
            lower_bound = mean_values - margin_of_error
            upper_bound = mean_values + margin_of_error
            # Create a DataFrame for the confidence intervals
            confidence_intervals = pd.DataFrame({'Lower Bound': lower_bound, 'Upper Bound': upper_bound})
            # Add confidence intervals to dictionary
            conf_interval_dict[day][device_id] = confidence_intervals.to_numpy()
    return conf_interval_dict

def flag(conf_interval_dict, speed_table):
    '''Flag values if they're outside the confidence interval range. Returns a nested dictionary {#day : {sensor_id : np.array(288,)}}'''
    flag_dict = {}
    num_days = speed_table[0][1].shape[0]
    num_time_points = speed_table[0][1].shape[1]
    for idx, day in enumerate(range(7, num_days+1)):
        flag_dict[day] = {}
        for device_id, speed_matrix in speed_table:
            new_row = []
            for time in range(num_time_points):
                speed = speed_matrix.iloc[idx, time]
                interval = conf_interval_dict[day][device_id][time]
                if speed >= interval[0] and speed <= interval[1]: # Flagging logic
                    new_row.append(0)
                else: # If value is outside threshold, flag it
                    new_row.append(1)
            flag_dict[day][device_id] = np.array(new_row)
    return flag_dict


directory = 'DelDOT Data'
speed_table = get_speed_from_csv(directory)

conf_interval_dict = calc_conf_interval({}, speed_table) # Nested dictionary. {#day : {sensor_id : np.array(288,2)}} np.array contains top and bottom threshold values
# for day_key, day_value in conf_interval_dict.items():
#     print(f"Day key: {day_key}")
#     for sensor_key, data in day_value.items():
#         print(f"  Sensor: {sensor_key}")
#         print(f"  Data: {data}")

flag_dict = flag(conf_interval_dict, speed_table)
# for day_key, day_value in flag_dict.items():
#     print(f"Day key: {day_key}")
#     for sensor_key, flags in day_value.items():
#         print(f"  Sensor: {sensor_key}")
#         print(f"  Flags: {flags}")
#         print(flags.shape)

'''Record confidence interval dictionary'''
with open('Flagged Data/confidence_intervals.pkl', 'wb') as file:
    pickle.dump(conf_interval_dict, file)

'''Record flagged data dictionary'''
with open('Flagged Data/flagged_data.pkl', 'wb') as file:
    pickle.dump(flag_dict, file)