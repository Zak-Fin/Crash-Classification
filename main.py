import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

raw_data_features = None
raw_data_labels = None
interpolated_timestamps = None
dir_name='cycle-crash-dataset/combined'
sessions = set()
# Categorize files containing different sensor sensor data
file_dict = dict()
# List of different activity names
activities = set()
file_names = os.listdir('cycle-crash-dataset/combined')
bar_cycle = []
gyro_cycle = []
accel_cycle = []
bar_crash = []
gyro_crash = []
accel_crash = []

for file_name in file_names:
    if '.txt' in file_name:
        tokens = file_name.split('_')
        identifier = (tokens[0])
        activity = tokens[2]
        sessions.add((identifier, activity))
        # Load the data from the file into a pandas dataframe
        data = pd.read_csv(os.path.join('cycle-crash-dataset/combined', file_name), header=None, sep=' ')

        # Preprocess the data (e.g., remove NaNs, apply filters, etc.)
        # ...
        accel = data.drop_duplicates(data.columns[0], keep='first').values
        # Spine-line interpolataion for x, y, z values (sampling rate is 32Hz).
        # Remove data in the first and last 3 seconds.
        timestamps = np.arange(accel[0, 0] + 3000.0, accel[-1, 0] - 3000.0, 1000.0 / 32)

        # Concatenate the preprocessed data with the appropriate list based on the sensor type and activity
        if identifier == 'barometer':
            if activity == 'cycle':
                bar_cycle.append(data)
            elif activity == 'crash':
                bar_crash.append(data)
        elif identifier == 'gyroscope':
            if activity == 'cycle':
                gyro_cycle.append(data)
            elif activity == 'crash':
                gyro_crash.append(data)
        elif identifier == 'accelerometer':
            if activity == 'cycle':
                accel_cycle.append(data)
            elif activity == 'crash':
                accel_crash.append(data)