import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
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


        if identifier == 'pressure':
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

def get_accel_gyro(data):

    accel_data = []
    for item in data:
        with open('cycle-crash-dataset/combined/'+item) as file:
            for line in file:
                pattern = r'X(-?\d+\.\d+)Y(-?\d+\.\d+)Z(-?\d+\.\d+)'
                string = line
                match = re.search(pattern, string)
                if match:
                    x = float(match.group(1))
                    y = float(match.group(2))
                    z = float(match.group(3))
                    accel_data.append([x, y, z])
    accel_array = np.array(accel_data)
    return accel_array

def get_bar(data):
    pattern = r'P(-?\d+\.\d+)'
    pressure_list =[]
    for item in data:
        with open('cycle-crash-dataset/combined/'+item) as file:
            for line in file:
                match = re.search(pattern, line)
                if match:
                    pressure_list.append(float(match.group(1)))
    return(np.array(pressure_list))



def get_features(accel,bar,gyro):

    accel_mean = np.mean(accel, axis=0)
    accel_std = np.std(accel, axis=0)
    accel_max = np.max(np.sum(np.square(accel), axis=1))
    print(accel_mean)
    bar_mean = np.mean(bar, axis=0)
    bar_std = np.std(bar, axis=0)
    # bar_max = np.max(np.sum(np.square(bar), axis=1))

    gyro_mean = np.mean(gyro, axis=0)
    gyro_std = np.std(gyro, axis=0)
    gyro_max = np.max(np.sum(np.square(gyro), axis=1))

    # Return the calculated features as a dictionary
    features = {'accel_mean': accel_mean,
                'accel_std': accel_std,
                'accel_max': accel_max,
                'bar_mean': bar_mean,
                'bar_std': bar_std,
                'gyro_mean': gyro_mean,
                'gyro_std': gyro_std,
                'gyro_max': gyro_max,}

    return features

def main():
    a = get_accel_gyro(accel_cycle)
    b = get_accel_gyro(gyro_cycle)
    c = get_bar(bar_cycle)


    features = get_features(a,c,b)
    print(features)


    # Extract features from the dictionary
    accel_mean = features['accel_mean']
    accel_std = features['accel_std']
    accel_max = features['accel_max']
    bar_mean = features['bar_mean']
    bar_std = features['bar_std']
    gyro_mean = features['gyro_mean']
    gyro_std = features['gyro_std']
    gyro_max = features['gyro_max']

    # Create a feature vector by concatenating the individual features
    feature_vector = np.array([accel_mean, accel_std, accel_max, bar_mean, bar_std, gyro_mean, gyro_std, gyro_max])

    # X = [feature_vector]

    # y = []

    # clf = RandomForestClassifier(n_estimators=100)
    # clf.fit(X, y)

    # # Use the classifier to predict the label for the new sample
    # label = clf.predict([new_feature_vector])[0]

    # sns.heatmap(feature_vector)
main()