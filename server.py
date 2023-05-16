# Server file for bike crash/cycle classification
# Hosted by www.pythonanywhere.com
# Live @ http://finmead.pythonanywhere.com/

from flask import Flask,request,jsonify
import pickle
app = Flask(__name__)
dir = 'cycle-crash-dataset/combined/'
import numpy as np
import re
import pandas as pd

# Default values for global vars
global state
state = ""
global html
html = f'''<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="refresh" content="1">
        <title>Bikerino</title>
        <meta name="description" content="Bikerino bike crash detection.">
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }}
            .bold-box {{
                background-color: #f5f5f5;
                padding: 20px;
                font-weight: bold;
                font-size: 1.2em;
                border-radius: 5px;
            }}
            .footer {{
                background-color: #333;
                color: #fff;
                padding: 24px;
                text-align: center;
                position: absolute;
                bottom: 0;
                width: 100%;
            }}
            .title {{
                font-size: 2em;
            }}
        </style>
    </head>
    <body>
        <h1 class="title">Bikerino bike crash detection</h1>
        <div class="bold-box">
            Classify smartphone accelerometer data as cycle or crash: <br>Current state: {state}
        </div>
        <div class="footer">
            &copy; zak-fin 2023
        </div>
    </body>
</html>'''

# Get the serialised classifier file (random-forest)
clf_path = "/home/finmead/mysite/classifier_2.pickle"
model = pickle.load(open(clf_path,'rb'))

# Extract the features from the sampled acceleromater data
def get_features(acc):
    data = []
    x_data = []
    y_data = []
    z_data = []

    # Split by commas and add x,y,z values for each entry to respective lists
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

    # calculate features for each value in lists
    features = {}
    features['mean_x'] = np.mean(x)
    features['mean_y'] = np.mean(y)
    features['mean_z'] = np.mean(z)
    features['std_x'] = np.std(x)
    features['std_y'] = np.std(y)
    features['std_z'] = np.std(z)
    features['label'] = 'no label'

    return features

@app.route('/')
def index():
    return html

@app.route('/predict',methods=['POST'])
def predict():
    data = request.json.get('data', [])
    # print(data)
    features = []
    features.append(get_features(data))
    df = pd.DataFrame(features)
    X = df.drop('label', axis=1)
    y = df['label']

    result = model.predict(X)
    # print(result)

    # Update html code with new classifcation result
    global state
    state = result[0]
    global html
    html = f'''<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="refresh" content="1">
        <title>Bikerino</title>
        <meta name="description" content="Bikerino bike crash detection.">
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }}
            .bold-box {{
                background-color: #f5f5f5;
                padding: 20px;
                font-weight: bold;
                font-size: 1.2em;
                border-radius: 5px;
            }}
            .footer {{
                background-color: #333;
                color: #fff;
                padding: 24px;
                text-align: center;
                position: absolute;
                bottom: 0;
                width: 100%;
            }}
            .title {{
                font-size: 2em;
            }}
        </style>
    </head>
    <body>
        <h1 class="title">Bikerino bike crash detection</h1>
        <div class="bold-box">
            Classify smartphone accelerometer data as cycle or crash: <br>Current state: <strong>{state}</strong>
        </div>
        <div class="footer">
            &copy; zak-fin 2023
        </div>
    </body>
</html>'''

    return jsonify({'result':str(result)})

# if __name__ == '__main__':
#     app.run(debug=True,host='0.0.0.0')