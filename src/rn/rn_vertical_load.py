# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
from keras.models import model_from_json
from keras.models import load_model
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# Importing the dataset
dataset = pd.read_csv('/home/jerson/catkin_ws/src/lidar_samples_reloaded/datasets/vert.csv')
X = dataset.iloc[:, 2: 182].values
y = dataset.iloc[:, -1].values
y = (y == ' true')

print(X)

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


###
    
# Fitting classifier to the Training set
# Create your classifier here
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input

#classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Predicting the Test set results


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

# import joblib
# filename = 'lidar_samples_core/src/modelSave/rn_torres_horiz.sav'
# joblib.dump(classifier, filename)
import joblib

#filename = '/home/mathias/catkin_ws/src/lidar_samples_reloaded/src/cnn/modelSave/model_torres_horiz.sav'
filename = '/home/jerson/catkin_ws/src/lidar_samples_reloaded/src/rn/lidar_samples_core/src/modelSave/rn_torres_vert.h5'
filenameSC = '/home/jerson/catkin_ws/src/lidar_samples_reloaded/src/cnn/sc_torres_vert.sav'

classifier= keras.models.load_model(filename)

sc = joblib.load(filenameSC)

# classifier = load_model('lidar_samples_core/src/modelSave/rn_torres_horiz.h5')

#print("Melhores Parâmetros:")
#print(best_param)
#print("Melhor accuracy: ")
#print(best_accur)
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

#Evaluating
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
print("--------------------")
cm = confusion_matrix(y_test, y_pred)

print(cm)
# ###


from sklearn.metrics import accuracy_score
"""
print(accuracy_score(y_test, y_pred)*100)
print("--------------------")



accuracies=[]
print("FSHH")
dataset = pd.read_csv('/home/mathias/catkin_ws/src/lidar_samples_reloaded/datasets/FSHH.csv')
X = dataset.iloc[:, 2: 182].values
y = dataset.iloc[:, -1].values
X = sc.transform(X) 
y_pred = classifier.predict(X)
accuracies.append(accuracy_score(y, y_pred)*100)




print("FSHA")
dataset = pd.read_csv('/home/mathias/catkin_ws/src/lidar_samples_reloaded/datasets/FSHA.csv')
X = dataset.iloc[:, 2: 182].values
y = dataset.iloc[:, -1].values
X = sc.transform(X) 
y_pred = classifier.predict(X)
accuracies.append(accuracy_score(y, y_pred)*100)

print("FSHT")
dataset = pd.read_csv('/home/mathias/catkin_ws/src/lidar_samples_reloaded/datasets/FSHT.csv')
X = dataset.iloc[:, 2: 182].values
y = dataset.iloc[:, -1].values
X = sc.transform(X) 
y_pred = classifier.predict(X)
accuracies.append(accuracy_score(y, y_pred)*100)

print("FSHTA")
dataset = pd.read_csv('/home/mathias/catkin_ws/src/lidar_samples_reloaded/datasets/FSHTA.csv')
X = dataset.iloc[:, 2: 182].values
y = dataset.iloc[:, -1].values
X = sc.transform(X) 
y_pred = classifier.predict(X)
accuracies.append(accuracy_score(y, y_pred)*100)

print("FSHTH")
dataset = pd.read_csv('/home/mathias/catkin_ws/src/lidar_samples_reloaded/datasets/FSHTH.csv')
X = dataset.iloc[:, 2: 182].values
y = dataset.iloc[:, -1].values
X = sc.transform(X) 
y_pred = classifier.predict(X)
accuracies.append(accuracy_score(y, y_pred)*100)

print("FSHTC")
dataset = pd.read_csv('/home/mathias/catkin_ws/src/lidar_samples_reloaded/datasets/FSHTC.csv')
X = dataset.iloc[:, 2: 182].values
y = dataset.iloc[:, -1].values
X = sc.transform(X) 
y_pred =	 classifier.predict(X)
accuracies.append(accuracy_score(y, y_pred)*100)

print("Approach")
dataset = pd.read_csv('/home/mathias/catkin_ws/src/lidar_samples_reloaded/datasets/approachH.csv')
X = dataset.iloc[:, 2: 182].values
y = dataset.iloc[:, -1].values
X = sc.transform(X) 
y_pred = classifier.predict(X)
accuracies.append(accuracy_score(y, y_pred)*100)
"""
final_times=[]
for i in range(1):
        print(i)
        for j in range(9999):
                X_FW = np.array(X[j])
                X_FW= np.reshape(X_FW, (1,180))
                start_time=time.time()
                y_pred = classifier.predict(X_FW)
#               final_times.append( time.time()-start_time)
                print("%s seconds" % (time.time() - start_time))


# final_times=[]
# for i in range(50):
# #	print(i)
# 	for j in range(200):
# 		X_FW = np.array(X[j])
# 		X_FW= np.reshape(X_FW, (1,180))
# 		start_time=time.time()
# 		y_pred = classifier.predict(X)
# 		final_times.append( time.time()-start_time)
# #		print("--- %s seconds ---" % (time.time() - start_time))


# print(final_times)

# n=np.array(final_times)
# print("Media= ", np.mean(n))
# print("Maior= ", np.amax(n))
# print("Menor= ", np.amin(n))
# print("Desvio Padrão= ",np.std(n))
# print("Variância= ",   np.var(n))
# """
# import matplotlib
# import matplotlib.pyplot as plt


# fig, ax = plt.subplots()
# torres = ('Only houses',  'Many trees', 'Only tower', 'Tower and  trees', 'Tower and houses','Tower and power plant', 'Approach')
# y_pos = np.arange(len(torres))
# hbars = ax.barh(y_pos, accuracies, align='center')
# ax.set_yticks(y_pos)
# ax.set_yticklabels(torres)
# ax.invert_yaxis()  # labels read top-to-bottom
# ax.set_xlabel('Accuracy (%)')
# ax.set_title('Performance in different scenarios')

# # Label with given captions, custom padding and annotate options
# #ax.bar_label(hbars, labels=['±%.2f' % e for e in error],
# #             padding=8, color='b', fontsize=14)
# ax.set_xlim(left=60,right=100)

# plt.plot()
# plt.show()
# print(accuracies)
# cm = confusion_matrix(y, y_pred)
# """
