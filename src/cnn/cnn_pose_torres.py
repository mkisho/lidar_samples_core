# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import math
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# Importing the dataset
dataset = pd.read_csv('/home/mathias/catkin_ws/src/lidar_samples_reloaded/datasets/DATASETTORRE.csv')
X = dataset.iloc[:, 2: 722].values
y = dataset.iloc[:, :2].values


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
X_train= np.reshape(X_train, (7999, 720, 1, 1))
X_test= np.reshape(X_test, (2000, 720, 1, 1))


###
    
# Fitting classifier to the Training set
# Create your classifier here
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
import tensorflow as tf

#classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Predicting the Test set results


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

#def build_model():
#    model = Sequential()
#    classifier.add(Dense(output_dim = 90, init = 'uniform', activation = 'relu', input_dim= 180))
#    classifier.add(Dense(output_dim = 20, init = 'uniform', activation = 'relu', input_dim= 180))
#    model.add(Input(shape=(720,)))
#    model.add(Dense(units = 200, activation = 'relu'))
#    model.add(Dense(units = 200, activation = 'relu'))
#    model.add(Dense(units = 100, activation = 'relu'))
#    model.add(Dense(units = 20, activation = 'relu'))
#    model.add(Dense(units = 2, activation = 'linear'))
#    model.output_shape
#    optimizer = tf.keras.optimizers.RMSprop(0.001)
#    model.compile(optimizer = optimizer, loss = 'mse', metrics = ['mae','mse'])
#    return model

#model=build_model()

#accuracies = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 10, n_jobs = -1)

#model.fit(X_train, y_train, epochs=100)

model = Sequential()
model.add(Conv2D(filters= 1, kernel_size= (5,1), activation='relu', input_shape=(720, 1, 1)))
model.add(MaxPooling2D((3,1)))
model.add(Dropout(0.5))

#model.add(Conv2D(filters=64, kernel_size= (1,5), activation='relu'))
#model.add(layers.MaxPooling2D((2, 1)))   
#model.add(layers.Conv2D(filters= 64, kernel_size= (3, 1), activation='relu'))

model.summary()
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(units = 2, activation = 'linear'))

model.summary()

#model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001), loss='mse', metrics=['mae','mse'])

model.fit(X_train, y_train, epochs=100, verbose=1, batch_size = 32)



#print("Melhores Parâmetros:")
#print(best_param)
#print("Melhor accuracy: ")
#print(best_accur)
y_pred = model.predict(X_test)
#y_pred = (y_pred>0.5)
t=[y_pred[:,0] - y_test[:,0],y_pred[:,1] - y_test[:,1]]


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
mse_x = mean_squared_error(y_pred[:,0], y_test[:,0])
mse_y = mean_squared_error(y_pred[:,1], y_test[:,1])
mae_x = mean_absolute_error(y_pred[:,0], y_test[:,0])
mae_y = mean_absolute_error(y_pred[:,1], y_test[:,1])

np.max(t[0])
df = pd.DataFrame([y_pred[:,0], y_test[:,0], y_pred[:,0] - y_test[:,0], y_pred[:,1], y_test[:,1], y_pred[:,1] - y_test[:,1]])


a= np.sort(y_pred[:,0] - y_test[:,0])

dft=df.T

dft.to_csv('/home/mathias/catkin_ws/src/lidar_samples_reloaded/datasets/Results.csv',header=["X pred", "X real", "X Diff","Y pred", "Y real", "Y Diff"])


mae=[]
mse=[]


print("FSHT")
dataset = pd.read_csv('/home/mathias/catkin_ws/src/lidar_samples_reloaded/datasets/FSHT.csv')
X = dataset.iloc[:, 2: 722].values
y = dataset.iloc[:, :2].values
X = sc.transform(X) 
X = np.reshape(X, (2499, 720, 1, 1))
y_pred = model.predict(X)
Y1=y_pred[:, 0]
X1=y_pred[:, 1]
Y2=y[:, 0]
X2=y[:, 1]
print([Y1,Y2])
A1=np.array([math.degrees(math.atan2(Y1[i],X1[i])) for i in range(len(Y1))])
A2=np.array([math.degrees(math.atan2(Y2[i],X2[i])) for i in range(len(Y2))])
mae.append(sum(abs(A1-A2))/len(A1))
mse.append(sum((A1-A2)**2)/len(A1))


print("FSHTA")
dataset = pd.read_csv('/home/mathias/catkin_ws/src/lidar_samples_reloaded/datasets/FSHTA.csv')
X = dataset.iloc[:, 2: 722].values
y = dataset.iloc[:, :2].values
X = sc.transform(X) 
X = np.reshape(X, (2499, 720, 1, 1))
y_pred = model.predict(X)
Y1=y_pred[:, 0]
X1=y_pred[:, 1]
Y2=y[:, 0]
X2=y[:, 1]
A1=np.array([math.degrees(math.atan2(Y1[i],X1[i])) for i in range(len(Y1))])
A2=np.array([math.degrees(math.atan2(Y2[i],X2[i])) for i in range(len(Y2))])
mae.append(sum(abs(A1-A2))/len(A1))
mse.append(sum((A1-A2)**2)/len(A1))


print("FSHTH")
dataset = pd.read_csv('/home/mathias/catkin_ws/src/lidar_samples_reloaded/datasets/FSHTH.csv')
X = dataset.iloc[:, 2: 722].values
y = dataset.iloc[:, :2].values
X = sc.transform(X) 
X = np.reshape(X, (2499, 720, 1, 1))
y_pred = model.predict(X)
Y1=y_pred[:, 0]
X1=y_pred[:, 1]
Y2=y[:, 0]
X2=y[:, 1]
A1=np.array([math.degrees(math.atan2(Y1[i],X1[i])) for i in range(len(Y1))])
A2=np.array([math.degrees(math.atan2(Y2[i],X2[i])) for i in range(len(Y2))])
mae.append(sum(abs(A1-A2))/len(A1))
mse.append(sum((A1-A2)**2)/len(A1))


print("FSHTC")
dataset = pd.read_csv('/home/mathias/catkin_ws/src/lidar_samples_reloaded/datasets/FSHTC.csv')
X = dataset.iloc[:, 2: 722].values
y = dataset.iloc[:, :2].values
X = sc.transform(X) 
X = np.reshape(X, (2499, 720, 1, 1))
y_pred = model.predict(X)
Y1=y_pred[:, 0]
X1=y_pred[:, 1]
Y2=y[:, 0]
X2=y[:, 1]
A1=np.array([math.degrees(math.atan2(Y1[i],X1[i])) for i in range(len(Y1))])
A2=np.array([math.degrees(math.atan2(Y2[i],X2[i])) for i in range(len(Y2))])
mae.append(sum(abs(A1-A2))/len(A1))
mse.append(sum((A1-A2)**2)/len(A1))

print("Approach")
dataset = pd.read_csv('/home/mathias/catkin_ws/src/lidar_samples_reloaded/datasets/approachH.csv')
X = dataset.iloc[:, 2: 722].values
y = dataset.iloc[:, :2].values
X = sc.transform(X) 
X = np.reshape(X, (200, 720, 1, 1))
y_pred = model.predict(X)
Y1=y_pred[:, 0]
X1=y_pred[:, 1]
Y2=y[:, 0]
X2=y[:, 1]
print([Y1,Y2])
print("x1=", X1)
print("X2=", X2)
A1=np.array([math.degrees(math.atan2(Y1[i],X1[i])) for i in range(len(Y1))])
A2=np.array([math.degrees(math.atan2(Y2[i],X2[i])) for i in range(len(Y2))])
#A1=np.array([math.sqrt(Y1[i]**2+X1[i]**2) for i in range(len(Y1))])
#A2=np.array([math.sqrt(Y2[i]**2+X2[i]**2) for i in range(len(Y2))])
print(A2, Y2, X2)

mae.append(sum(abs(A1-A2))/len(A1))
mse.append(sum((A1-A2)**2)/len(A1))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


labels = ['Only \nTower', 'Tower and\ntwo trees', 'Tower and\n houses','Tower and\n many trees','Approach']

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
#rects1 = ax.bar(x, mae, width, label='Mean absolute error in angle prediction', align='center')
rects1 = ax.bar(x - width/2, mae, width, label='Mean Absolute Error')
rects2 = ax.bar(x + width/2, mse, width, label='Mean Squared Error')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('')
ax.set_title('Performance in different scenarios')
ax.set_xticks(x)
ax.set_xticklabels(labels,fontsize=8)
ax.legend()

#ax.bar_label(rects1, padding=3)
#ax.bar_label(rects2, padding=3)

fig.tight_layout()
print(mae)
print(mse)
plt.show()

#model.save("model")

#Evaluating
# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)

#print(cm)
###

df = pd.DataFrame([A2[:], A1[:]])
dft=df.T
mse = mean_squared_error(A1, A2)
mae = mean_absolute_error(A1, A2)
print("mae= ", mae)
print("mse= ", mse)
dft.to_csv('/home/mathias/catkin_ws/src/lidar_samples_reloaded/datasets/approachRH.csv',header=["X true", "X pred"])
