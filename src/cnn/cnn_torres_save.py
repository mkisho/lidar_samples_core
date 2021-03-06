# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import tensorflow as tf
from keras.models import model_from_json
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# Importing the dataset
dataset = pd.read_csv('/home/mathias/catkin_ws/src/lidar_samples_reloaded/datasets/horiz.csv')
X = dataset.iloc[:, 2: 722].values
y = dataset.iloc[:, -1].values
print(y)
y = (y == ' true')
#y = y.astype(int)

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test)


X_train= np.reshape(X_train, (len(X_train), 720, 1, 1))
X_test= np.reshape(X_test, (len(X_test), 720, 1, 1))


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

#classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Predicting the Test set results


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

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
model.add(Dense(units = 1, activation = 'sigmoid'))

model.summary()

#model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100, verbose=1, batch_size = 32)
model.save('cnn_torres_horiz.h5')
import joblib

#filename = '/home/mathias/catkin_ws/src/lidar_samples_reloaded/src/cnn/modelSave/model_torres_horiz.sav'
filenameSC = '/home/mathias/catkin_ws/src/lidar_samples_reloaded/src/cnn/sc_torres_horiz.sav'
#joblib.dump(sv, filename)
joblib.dump(sc, filenameSC)


#classifier.save("classificador")

#print("Melhores Par??metros:")
#print(best_param)
#print("Melhor accuracy: ")
#print(best_accur)
y_pred = model.predict(X_test)
y_pred = (y_pred>0.5)

#Evaluating
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
print("--------------------")
cm = confusion_matrix(y_test, y_pred)

print(cm)
###
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred)*100)
print("--------------------")



accuracies=[]
print("FSHH")
dataset = pd.read_csv('/home/mathias/catkin_ws/src/lidar_samples_reloaded/datasets/FSHH.csv')
X = dataset.iloc[:, 2: 722].values
y = dataset.iloc[:, -1].values
X = sc.transform(X) 
X= np.reshape(X, (2499, 720, 1, 1))
y_pred = model.predict(X)
y_pred = (y_pred>0.5)
y = (y == ' true')
print(y)
print(y_pred)
accuracies.append(accuracy_score(y, y_pred)*100)




print("FSHA")
dataset = pd.read_csv('/home/mathias/catkin_ws/src/lidar_samples_reloaded/datasets/FSHA.csv')
X = dataset.iloc[:, 2: 722].values
y = dataset.iloc[:, -1].values
X = sc.transform(X) 
X= np.reshape(X, (2499, 720, 1, 1))
y_pred = model.predict(X)
y_pred = (y_pred>0.5)
y = (y == ' true')
accuracies.append(accuracy_score(y, y_pred)*100)


print("FSHT")
dataset = pd.read_csv('/home/mathias/catkin_ws/src/lidar_samples_reloaded/datasets/FSHT.csv')
X = dataset.iloc[:, 2: 722].values
y = dataset.iloc[:, -1].values
X = sc.transform(X) 
X= np.reshape(X, (2499, 720, 1, 1))
y_pred = model.predict(X)
y_pred = (y_pred>0.5)
y = (y == ' true')
accuracies.append(accuracy_score(y, y_pred)*100)

print("FSHTA")
dataset = pd.read_csv('/home/mathias/catkin_ws/src/lidar_samples_reloaded/datasets/FSHTA.csv')
X = dataset.iloc[:, 2: 722].values
y = dataset.iloc[:, -1].values
X = sc.transform(X) 
X= np.reshape(X, (2499, 720, 1, 1))
y_pred = model.predict(X)
y = (y == ' true')
y_pred = (y_pred>0.5)
accuracies.append(accuracy_score(y, y_pred)*100)

print("FSHTH")
dataset = pd.read_csv('/home/mathias/catkin_ws/src/lidar_samples_reloaded/datasets/FSHTH.csv')
X = dataset.iloc[:, 2: 722].values
y = dataset.iloc[:, -1].values
X = sc.transform(X) 
X= np.reshape(X, (2499, 720, 1, 1))
y_pred = model.predict(X)
y = (y == ' true')
y_pred = (y_pred>0.5)
accuracies.append(accuracy_score(y, y_pred)*100)

print("FSHTC")
dataset = pd.read_csv('/home/mathias/catkin_ws/src/lidar_samples_reloaded/datasets/FSHTC.csv')
X = dataset.iloc[:, 2: 722].values
y = dataset.iloc[:, -1].values
X = sc.transform(X) 
X= np.reshape(X, (2499, 720, 1, 1))
y_pred =model.predict(X)
y = (y == ' true')
y_pred = (y_pred>0.5)
accuracies.append(accuracy_score(y, y_pred)*100)

print("Approach")
dataset = pd.read_csv('/home/mathias/catkin_ws/src/lidar_samples_reloaded/datasets/approachH.csv')
X = dataset.iloc[:, 2: 722].values
y = dataset.iloc[:, -1].values
X = sc.transform(X) 
X= np.reshape(X, (200, 720, 1, 1))
y_pred = model.predict(X)
y = (y == ' true')
y_pred = (y_pred>0.5)
accuracies.append(accuracy_score(y, y_pred)*100)



#final_times=[]
#for i in range(50):
##	print(i)
#	for j in range(200):
#		X_FW = np.array(X[j])
#		X_FW= np.reshape(X_FW, (1,720))
#		start_time=time.time()
#		y_pred = model.predict(X)
#		final_times.append( time.time()-start_time)
##		print("--- %s seconds ---" % (time.time() - start_time))


#print(final_times)

#n=np.array(final_times)
#print("Media= ", np.mean(n))
#print("Maior= ", np.amax(n))
#print("Menor= ", np.amin(n))
#print("Desvio Padr??o= ",np.std(n))
#print("Vari??ncia= ",   np.var(n))

import matplotlib
import matplotlib.pyplot as plt


fig, ax = plt.subplots()
torres = ('Only houses',  'Many trees', 'Only tower', 'Tower and  trees', 'Tower and houses','Tower and power plant', 'Approach')
y_pos = np.arange(len(torres))
hbars = ax.barh(y_pos, accuracies, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(torres)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Accuracy (%)')
ax.set_title('Performance in different scenarios')

# Label with given captions, custom padding and annotate options
#ax.bar_label(hbars, labels=['??%.2f' % e for e in error],
#             padding=8, color='b', fontsize=14)
ax.set_xlim(left=60,right=100)

plt.plot()
plt.show()
print(accuracies)
cm = confusion_matrix(y, y_pred)


