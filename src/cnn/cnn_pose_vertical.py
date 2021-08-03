# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import math
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# Importing the dataset
dataset = pd.read_csv('/home/mathias/catkin_ws/src/lidar_samples_reloaded/datasets/DATASETVERTICALTORRE.csv')
X = dataset.iloc[:, 2: 182].values
dist = dataset.iloc[:, :2].values
y= np.array([math.sqrt(dist[i][0]**2+dist[i][1]**2) for i in range(0,9999)])

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
X_train= np.reshape(X_train, (7999, 180, 1, 1))
X_test= np.reshape(X_test, (2000, 180, 1, 1))


###
    
# Fitting classifier to the Training set
# Create your classifier here
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input
import tensorflow as tf

#classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Predicting the Test set results


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout


model = Sequential()
model.add(Conv2D(filters= 64, kernel_size= (9,1), activation='relu', input_shape=(180, 1, 1)))
model.add(MaxPooling2D((9,1)))
model.add(Conv2D(filters= 64, kernel_size= (3,1), activation='relu'))
model.add(MaxPooling2D((3,1)))
model.add(Conv2D(filters= 64, kernel_size= (3,1), activation='relu'))
model.add(MaxPooling2D((3,1)))

#model.add(Conv2D(filters=64, kernel_size= (1,5), activation='relu'))
#model.add(layers.MaxPooling2D((2, 1)))   
#model.add(layers.Conv2D(filters= 64, kernel_size= (3, 1), activation='relu'))

model.summary()
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(units = 1, activation = 'linear'))

model.summary()

#model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
#model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001), loss='mse', metrics=['mae','mse'])
model.compile(optimizer='adam', loss='mse', metrics=['mae','mse'])

model.fit(X_train, y_train, epochs=100, verbose=1, batch_size = 32)

y_pred = np.array(model.predict(X_test))
t=[y_pred, y_test ,y_pred - y_test]


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
mse = mean_squared_error(y_pred, y_test)
mae = mean_absolute_error(y_pred, y_test)

y_pred=y_pred.flatten()
np.max(t[0])
df = pd.DataFrame([y_pred[:], y_test[:], y_pred[:] - y_test[:]])
a= np.sort(y_pred - y_test)
print(mae)
print(mse)



dft=df.T

dft.to_csv('/home/mathias/catkin_ws/src/lidar_samples_reloaded/datasets/ResultsV.csv',header=["X pred", "X real", "X Diff"])




dataset = pd.read_csv('/home/mathias/catkin_ws/src/lidar_samples_reloaded/datasets/approachV.csv')
X = dataset.iloc[:, 2: 182].values
y = dataset.iloc[:, :2].values
X = sc.transform(X)
X= np.reshape(X, (799, 180, 1, 1))
y_pred = model.predict(X)

Y2=y[:, 0]
X2=y[:, 1]
y_true=[math.sqrt(X2[i]**2+Y2[i]**2) for i in range(len(Y2))]

df = pd.DataFrame([y_true[:], y_pred[:]])
dft=df.T
#mse = mean_squared_error(y_pred, y_true)
#mae = mean_absolute_error(y_pred, y_true)
#print("mae= ", mae)
#print("mse= ", mse)
#dft.to_csv('/home/mathias/catkin_ws/src/lidar_samples_reloaded/datasets/approachRV.csv',header=["X true", "X pred"])

