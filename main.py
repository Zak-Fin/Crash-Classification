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
bar_cycle = dict()
gyro_cycle = dict()
accel_cycle = dict()
bar_crash = dict()
gyro_crash = dict()
accel_crash = dict()

for file_name in file_names:
    if '.txt' in file_name:
        tokens = file_name.split('_')
        identifier = (tokens[0])
        activity = tokens[2]
        sessions.add((identifier, activity))
        # Load the data from the file into a pandas dataframe
        data = pd.read_csv(os.path.join('cycle-crash-dataset/combined', file_name), header=None, sep=' ')
        accel = data.drop_duplicates(data.columns[0], keep='first').values


        if identifier == 'barometer':
            if activity == 'cycle':
                bar_cycle[(file_name)]=(data)
            elif activity == 'crash':
                bar_crash[(file_name)]=(data)
        elif identifier == 'gyroscope':
            if activity == 'cycle':
                gyro_cycle[(file_name)]=(data)
            elif activity == 'crash':
                gyro_crash[(file_name)]=(data)
        elif identifier == 'accelerometer':
            if activity == 'cycle':
                accel_cycle[(file_name)]=(data)
            elif activity == 'crash':
                accel_crash[(file_name)]=(data)




for item in accel_cycle:
    with open('cycle-crash-dataset/combined/'+item) as file:
        for line in file:

            print(line)
