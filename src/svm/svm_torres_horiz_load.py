# Classification template

# Importing the libraries
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# Importing the dataset
dataset = pd.read_csv('/home/jerson/catkin_ws/src/lidar_samples_reloaded/datasets/horiz.csv')
X = dataset.iloc[:, 2: 722].values
y = dataset.iloc[:, -1].values

print(X)
print(y)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer 
#labelEncoder_X_1= LabelEncoder()
#X[:,1] = labelEncoder_X_1.fit_transform(X[:,1])
#labelEncoder_X_2= LabelEncoder()
#X[:,2] = labelEncoder_X_2.fit_transform(X[:,2])
#ct = ColumnTransformer([("OneHot", OneHotEncoder(),[1])], remainder="passthrough") 
#ct.fit_transform(X)    
#oneHotEncoder = OneHotEncoder(categorical_features = [1])
#X= oneHotEncoder.fit_transform(X).toarray()
#X=X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test)



import joblib

filename = '/home/jerson/catkin_ws/src/lidar_samples_reloaded/src/svm/modelSave/model_torres_horiz.sav'
filenameSC = '/home/jerson/catkin_ws/src/lidar_samples_reloaded/src/svm/modelSave/sc_torres_horiz.sav'
sc = joblib.load(filenameSC)
sv = joblib.load(filename)

# sv = joblib.load(filename)

#mensagen tipo laser scan vindo do topico /uav1/rplidar/scan

y_pred= sv.predict(X_test)


print("accuracy= ")
print (accuracy_score(y_test, y_pred)*100)


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
cm = confusion_matrix(y_test, y_pred)
print (cm)



#Evaluating
# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)

#print(cm)
###

import time
"""
accuracies=[]

print("All Scenarios")
dataset = pd.read_csv('/home/mathias/catkin_ws/src/lidar_samples_reloaded/datasets/FSHH.csv')
X = dataset.iloc[:, 2: 722].values
y = dataset.iloc[:, -1].values
X = sc.transform(X) 
y_pred = sv.predict(X)
accuracies.append(accuracy_score(y, y_pred)*100)

print("FSHA")
dataset = pd.read_csv('/home/mathias/catkin_ws/src/lidar_samples_reloaded/datasets/FSHA.csv')
X = dataset.iloc[:, 2: 722].values
y = dataset.iloc[:, -1].values
X = sc.transform(X) 
y_pred = sv.predict(X)
accuracies.append(accuracy_score(y, y_pred)*100)

print("FSHT")
dataset = pd.read_csv('/home/mathias/catkin_ws/src/lidar_samples_reloaded/datasets/FSHT.csv')
X = dataset.iloc[:, 2: 722].values
y = dataset.iloc[:, -1].values
X = sc.transform(X) 
y_pred = sv.predict(X)
accuracies.append(accuracy_score(y, y_pred)*100)

print("FSHTA")
dataset = pd.read_csv('/home/mathias/catkin_ws/src/lidar_samples_reloaded/datasets/FSHTA.csv')
X = dataset.iloc[:, 2: 722].values
y = dataset.iloc[:, -1].values
X = sc.transform(X) 
y_pred = sv.predict(X)
accuracies.append(accuracy_score(y, y_pred)*100)

print("FSHTH")
dataset = pd.read_csv('/home/mathias/catkin_ws/src/lidar_samples_reloaded/datasets/FSHTH.csv')
X = dataset.iloc[:, 2: 722].values
y = dataset.iloc[:, -1].values
X = sc.transform(X) 
y_pred = sv.predict(X)
accuracies.append(accuracy_score(y, y_pred)*100)

print("FSHTC")
dataset = pd.read_csv('/home/mathias/catkin_ws/src/lidar_samples_reloaded/datasets/FSHTC.csv')
X = dataset.iloc[:, 2: 722].values
y = dataset.iloc[:, -1].values
X = sc.transform(X) 
y_pred = sv.predict(X)
accuracies.append(accuracy_score(y, y_pred)*100)

print("Approach")
dataset = pd.read_csv('/home/mathias/catkin_ws/src/lidar_samples_reloaded/datasets/approachH.csv')
X = dataset.iloc[:, 2: 722].values
y = dataset.iloc[:, -1].values
X = sc.transform(X) 
y_pred = sv.predict(X)
accuracies.append(accuracy_score(y, y_pred)*100)


print(accuracies)
"""

final_times=[]
for i in range(1):
        print(i)
        for j in range(9999):
                X_FW = np.array(X[j])
                X_FW= np.reshape(X_FW, (1,720))
                start_time=time.time()
                y_pred = sv.predict(X_FW)
#               final_times.append( time.time()-start_time)
                print("%s seconds" % (time.time() - start_time))

