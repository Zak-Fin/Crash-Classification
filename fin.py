import os
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

file_names = os.listdir('cycle-crash-dataset/combined')

def extract_features(file_list):
    features = []
    labels = []

    accel_mean_feature = [0]
    accel_std_feature = [0]
    accel_max_feature = [0]
    bar_mean_feature = [0]
    bar_std_feature = [0]
    gyro_mean_feature = [0]
    gyro_std_feature = [0]
    gyro_max_feature = [0]


    for file in file_list:
        
        # file = pd.read_csv(os.path.join('cycle-crash-dataset/combined', file_name), header=None, sep=' ')
        # Get the label (cycle or crash) from the file name
        tokens = file.split('_')
        sensorType = tokens[0]
        label = tokens[2]
        print(label, sensorType)
        labels = np.append(labels, label)
        # labels.append(label)

        # Read the sensor data from the file
        with open('cycle-crash-dataset/combined/'+file, 'r') as f:
            data = [line.strip() for line in f if line.strip()] # remove empty lines

        # Initialize empty arrays for each sensor data
        gyro_data = []
        accel_data = []
        pressure_data = []
        print(data)
        # Extract the data for each sensor type
        for row in data:
            if sensorType == 'gyroscope':
                try:
                    x, y, z = row.split('X')[1].split('Y')[0], row.split('Y')[1].split('Z')[0], row.split('Z')[1]
                    gyro_data.append([float(x), float(y), float(z)])
                except IndexError:
                    continue
            elif sensorType == 'accelerometer':
                try:
                    x, y, z = row.split('X')[1].split('Y')[0], row.split('Y')[1].split('Z')[0], row.split('Z')[1]
                    accel_data.append([float(x), float(y), float(z)])
                except IndexError:
                    continue
            elif sensorType == 'pressure':
                try:
                    pressure = float(row.split('P')[1])
                    pressure_data.append(pressure)
                except IndexError:
                    continue
        
        

        # Calculate the mean, standard deviation and maximum values for each sensor data
        # Extract features from the data
            if sensorType == 'accelerometer':
                # Convert data from string to float
                # data = [[float(val) for val in line.strip().split('XYZ')] for line in data]

                # Compute mean, standard deviation, and maximum for each axis
                accel_means = np.mean(accel_data, axis=0)
                accel_stds = np.std(accel_data, axis=0)
                accel_maxes = np.max(accel_data, axis=0)

                # Concatenate features into a single vector
                accel_features = np.concatenate([accel_means, accel_stds, accel_maxes])

                # Add features and label to the lists
                features = np.append(features, accel_features)
                labels = np.append(labels, label)

            elif sensorType == 'gyroscope':
                # Convert data from string to float
                # data = [[float(val) for val in line.strip().split('XYZ')] for line in data]

                # Compute mean, standard deviation, and maximum for each axis
                gyro_means = np.mean(gyro_data, axis=0)
                gyro_stds = np.std(gyro_data, axis=0)
                gyro_maxes = np.max(gyro_data, axis=0)

                # Concatenate features into a single vector
                gyro_features = np.concatenate([gyro_means, gyro_stds, gyro_maxes])

                # Add features and label to the lists
                features = np.append(features, gyro_features)
                labels = np.append(labels, label)

            elif sensorType == 'pressure':
                # Convert data from string to float
                # data = [[float(line.strip()[1:])] for line in data]

                # Compute mean, standard deviation, and maximum
                pressure_mean = np.mean(pressure_data)
                pressure_std = np.std(pressure_data)
                pressure_max = np.max(pressure_data)

                # Concatenate features into a single vector
                pressure_features = np.array([pressure_mean, pressure_std, pressure_max]).flatten()

                # Add features and label to the lists
                features = np.append(features, pressure_features)
                labels = np.append(labels, label)
        
        # gyro_data = np.array(gyro_data)
        # accel_data = np.array(accel_data)
        # pressure_data = np.array(pressure_data)
        
        # gyro_mean = np.mean(gyro_data, axis=0)
        # gyro_std = np.std(gyro_data, axis=0)
        # gyro_max = np.max(gyro_data, axis=0)
        
        # accel_mean = np.mean(accel_data, axis=0)
        # accel_std = np.std(accel_data, axis=0)
        # accel_max = np.max(accel_data, axis=0)
        
        # pressure_mean = np.mean(pressure_data)
        # pressure_std = np.std(pressure_data)
        # pressure_max = np.max(pressure_data)
        
        # Append the feature vector to the list of features
        # feature_vector = np.concatenate((gyro_mean, gyro_std, gyro_max,
        #                                   accel_mean, accel_std, accel_max,
        #                                   pressure_mean, pressure_std, pressure_max))
        # features.append(feature_vector)
        # Convert features and labels to numpy arrays
        features = np.array(features)
        labels = np.array(labels)

    return np.array(features), np.array(labels)

if __name__ == "__main__":
    features, labels = extract_features(file_list=file_names)
    # accel_mean = features[0]
    # accel_std = features[1]
    # accel_max = features[2]
    # bar_mean = features[3]
    # bar_std = features[4]
    # gyro_mean = features[5]
    # gyro_std = features[6]
    # gyro_max = features[7]
    # print('accel_mean %f' % (accel_mean))
    # print('accel_std %f'% (accel_std))
    # print('accel_max %f'% (accel_max))
    # print('gyro_mean %f'% (gyro_mean))
    # print('gyro_std %f'% (gyro_std))
    # print('gyro_max %f'% (gyro_max))
    # print('bar_mean %f'% (bar_mean))
    # print('bar_std %f'% (bar_std))
    # np.set_printoptions(threshold=np.inf)
    # print(features)
    # assuming features and labels are 1D arrays
    # features = np.reshape(features, (-1, 1))
    # labels = np.reshape(labels, (-1, 1))

    # create the random forest classifier
    rfc = RandomForestClassifier(n_estimators=100)

    # fit the classifier to the data
    rfc.fit(features, labels)
