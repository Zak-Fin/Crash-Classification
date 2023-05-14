from flask import Flask,request,jsonify
import pickle
model = pickle.load(open('./output/classifier.pickle','rb'))
app = Flask(__name__)
dir = 'cycle-crash-dataset/combined/'
import numpy as np
# import os
import re
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix
# from sklearn.model_selection import train_test_split
import pandas as pd
# import ast

def extract_features_acc(file_name):
    data = []
    x_data = []
    y_data = []
    z_data = []
    tokens = file_name.split('_')
    identifier = (tokens[0])
    activity = tokens[2]
    with open(dir + file_name) as file:
        for line in file:
            pattern = r'X(-?\d+\.\d+)Y(-?\d+\.\d+)Z(-?\d+\.\d+)'
            string = line
            match = re.search(pattern, string)
            if match:
                x = float(match.group(1))
                y = float(match.group(2))
                z = float(match.group(3))
                x_data.append(x)
                y_data.append(y)
                z_data.append(z)
                # print(x,y,z)

        data = [x_data, y_data, z_data]

        x = data[0]
        y = data[1]
        z = data[2]

        # print(x)
        features = {}
        features['mean_x'] = np.mean(x)
        features['mean_y'] = np.mean(y)
        features['mean_z'] = np.mean(z)
        features['std_x'] = np.std(x)
        features['std_y'] = np.std(y)
        features['std_z'] = np.std(z)
        features['label'] = activity

        return features

def get_features(acc):
    data = []
    x_data = []
    y_data = []
    z_data = []

    sample = acc.split(',')

    for sensorValue in sample:
        print(sensorValue)
        pattern = r'X(-?\d+\.\d+)Y(-?\d+\.\d+)Z(-?\d+\.\d+)'
        match = re.search(pattern, sensorValue)
        if match:
            x = float(match.group(1))
            y = float(match.group(2))
            z = float(match.group(3))
            x_data.append(x)
            y_data.append(y)
            z_data.append(z)
            print(x,y,z)

    data = [x_data, y_data, z_data]

    x = data[0]
    y = data[1]
    z = data[2]

    features = {}
    features['mean_x'] = np.mean(x)
    features['mean_y'] = np.mean(y)
    features['mean_z'] = np.mean(z)
    features['std_x'] = np.std(x)
    features['std_y'] = np.std(y)
    features['std_z'] = np.std(z)
    features['label'] = 'what'

    return features

@app.route('/')
def index():
    print("beans")
    return "Hello world"

@app.route('/predict',methods=['POST'])
def predict():
    data = request.json.get('data', [])
    # coordinates_string = request.data.decode('utf-8')
    # print(data)
    features = []
    features.append(get_features(data))
    df = pd.DataFrame(features)
    X = df.drop('label', axis=1)
    y = df['label']

    result = model.predict(X)

    return jsonify({'result':str(result)})

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')