'''File for changing new DelDOT format to the old format because the codebase was originally designed for the old format in csv'''

import os
import pandas as pd

# Directory containing your CSV files
input_directory = 'DelDOT Data New'
output_directory = 'DelDOT Reformatted'

# Ensure the output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Function to process and reformat each file
def reformat_csv(file_path, output_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Select and rename the necessary columns
    df_reformatted = pd.DataFrame({
        'index': df['Date'],
        'device_id': df['Device ID'],
        'direction': df['Dir'].replace({'N': 'NB', 'S': 'SB', 'E': 'EB', 'W': 'WB'}),  # Example for direction mapping
        'flow_status': [1] * len(df),  # Assuming flow_status is always 1
        'speed': df['Avg Speed'],
        'volume': df['Volume'],
        'occupancy': df['Occupancy %'].str.rstrip('%').astype(float),  # Convert occupancy percentage to float
    })

    # Reset index and ensure unique column names
    df_reformatted.reset_index(inplace=True)
    if 'level_0' not in df_reformatted.columns:
        df_reformatted.insert(0, 'level_0', df_reformatted.index)

    # Save the updated CSV to output path
    df_reformatted.to_csv(output_path, index=False)

# Iterate over all CSV files in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith('.csv'):
        input_file_path = os.path.join(input_directory, filename)
        output_file_path = os.path.join(output_directory, filename)
        reformat_csv(input_file_path, output_file_path)

print("Reformatting complete!")