# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# Importing the dataset
dataset = pd.read_csv('/home/mathias/catkin_ws/src/lidar_samples_reloaded/datasets/dadosFinais.csv')
X = dataset.iloc[:, 2: 722].values
y = dataset.iloc[:, -1].values

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

from sklearn import svm
from sklearn.model_selection import GridSearchCV

parameters = {'C':[1, 2, 3], 'gamma':[0.01, 0.02]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(X_train, y_train)
sorted(clf.cv_results_.keys())

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
    % (mean, std * 2, params))

#from sklearn import svm
#clf = svm.SVC(C=2,gamma=0.01)
#clf.fit(X_train, y_train)
#y_pred= clf.predict(X_test)

#Evaluating
# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)

#print(cm)
###



accuracies=[]
print("FSHH")
dataset = pd.read_csv('/home/mathias/catkin_ws/src/lidar_samples_reloaded/datasets/FSHH.csv')
X = dataset.iloc[:, 2: 722].values
y = dataset.iloc[:, -1].values
X = sc.transform(X) 
y_pred = clf.predict(X)
accuracies.append(accuracy_score(y, y_pred)*100)




print("FSHA")
dataset = pd.read_csv('/home/mathias/catkin_ws/src/lidar_samples_reloaded/datasets/FSHA.csv')
X = dataset.iloc[:, 2: 722].values
y = dataset.iloc[:, -1].values
X = sc.transform(X) 
y_pred = clf.predict(X)
accuracies.append(accuracy_score(y, y_pred)*100)

print("FSHT")
dataset = pd.read_csv('/home/mathias/catkin_ws/src/lidar_samples_reloaded/datasets/FSHT.csv')
X = dataset.iloc[:, 2: 722].values
y = dataset.iloc[:, -1].values
X = sc.transform(X) 
y_pred = clf.predict(X)
accuracies.append(accuracy_score(y, y_pred)*100)

print("FSHTA")
dataset = pd.read_csv('/home/mathias/catkin_ws/src/lidar_samples_reloaded/datasets/FSHTA.csv')
X = dataset.iloc[:, 2: 722].values
y = dataset.iloc[:, -1].values
X = sc.transform(X) 
y_pred = clf.predict(X)
accuracies.append(accuracy_score(y, y_pred)*100)

print("FSHTH")
dataset = pd.read_csv('/home/mathias/catkin_ws/src/lidar_samples_reloaded/datasets/FSHTH.csv')
X = dataset.iloc[:, 2: 722].values
y = dataset.iloc[:, -1].values
X = sc.transform(X) 
y_pred = clf.predict(X)
accuracies.append(accuracy_score(y, y_pred)*100)

print("FSHTC")
dataset = pd.read_csv('/home/mathias/catkin_ws/src/lidar_samples_reloaded/datasets/FSHTC.csv')
X = dataset.iloc[:, 2: 722].values
y = dataset.iloc[:, -1].values
X = sc.transform(X) 
y_pred =	 clf.predict(X)
accuracies.append(accuracy_score(y, y_pred)*100)

print("Approach")
dataset = pd.read_csv('/home/mathias/catkin_ws/src/lidar_samples_reloaded/datasets/approachH.csv')
X = dataset.iloc[:, 2: 722].values
y = dataset.iloc[:, -1].values
X = sc.transform(X) 
y_pred = clf.predict(X)
accuracies.append(accuracy_score(y, y_pred)*100)

start_time=time.time()

for i in range(400):
	y_pred = clf.predict(X)

print("--- %s seconds ---" % (time.time() - start_time))


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
#ax.bar_label(hbars, labels=['±%.2f' % e for e in error],
#             padding=8, color='b', fontsize=14)
ax.set_xlim(left=60,right=100)

plt.plot()
plt.show()
print(accuracies)
cm = confusion_matrix(y, y_pred)