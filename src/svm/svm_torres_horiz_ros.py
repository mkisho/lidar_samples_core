# Classification template

# Importing the libraries

from pickle import FALSE
import joblib
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import rospy

from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool

from numpy import inf


# from pickle import load
# from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# from sklearn.compose import ColumnTransformer 

# Feature Scaling
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

filenameSC = '/home/mathias/catkin_ws/src/lidar_samples_reloaded/src/svm/modelSave/sc_torres_horiz.sav'
filename = '/home/mathias/catkin_ws/src/lidar_samples_reloaded/src/svm/modelSave/model_torres_horiz.sav'


sc = joblib.load(filenameSC)
sv = joblib.load(filename)

isTorre = False


def callback(data):
	if(len(data.ranges)==720):
		X = data.ranges
		X = np.reshape(X, (1,720))
		X[X == inf] = 100
		X_test = sc.transform(X)
		y_pred = sv.predict(X_test)
		isTorre = y_pred
		talker()
		print(y_pred)


def listener():

    rospy.init_node('detect_horiz', anonymous=True)

    rospy.Subscriber("/uav1/rplidar/scan", LaserScan, callback)

    rospy.spin()

def talker():
    pub = rospy.Publisher('/uav1/horiz', Bool, queue_size=1)
    pubTorre = isTorre
    pub.publish(pubTorre)



if __name__ == '__main__':
    
    try:
        listener()
        
    except rospy.ROSInterruptException:
        pass





