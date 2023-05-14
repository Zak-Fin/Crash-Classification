import math
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
dir = 'cycle-crash-dataset/combined/'
def extract_features_acc(file_name):
      data = []
      x_data = []
      y_data = []
      z_data = []
      tokens = file_name.split('_')
      identifier = (tokens[0])
      activity = tokens[2]
      with open(dir+file_name) as file:
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
      
def serialise_classifier(clf):
         with open('output/classifier.pickle', 'wb') as f:
             pickle.dump(clf, f)
   
if __name__ == "__main__":
    #  file_nam = 'accelerometer_data_cycle_20181114_121451.txt'
    #  print(extract_features_acc(file_nam))

     file_names = os.listdir('cycle-crash-dataset/combined')
     features = []
    
     for file_name in file_names:
         if 'accelerometer' in file_name:
             features.append(extract_features_acc(file_name))
            #  print(features)
        #  if 'gyroscope' in file_name:
        #      features.append(extract_features_acc(file_name))
    #  print(features)
    #  print(features[0]['mean_x'], features[1]['mean_x'])
     # Drop dictionaries with nan values
    #  features = [d for d in features if not any(isinstance(v, (int, float)) and math.isnan(v) for v in d.values())]

     df = pd.DataFrame(features)


     X = df.drop('label', axis=1)
     y = df['label']
    
     print(np.size(df))

     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)
     
     clf = RandomForestClassifier(n_estimators=100, random_state=41, max_depth=5)
     clf.fit(X_train, y_train)

     y_pred = clf.predict(X_test)
     # evaluate the performance of the model
     accuracy = accuracy_score(y_test, y_pred)
     print("Accuracy:", accuracy)

     serialise_classifier(clf=clf)

     # Create a confusion matrix
     cm = confusion_matrix(y_test, y_pred)

     # Plot the confusion matrix as a heatmap
     sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=['cycle', 'crash'], yticklabels=['cycle', 'crash'])

     plt.xlabel('Predicted')
     plt.ylabel('True')
     plt.title('RF - Confusion Matrix')
     plt.show()