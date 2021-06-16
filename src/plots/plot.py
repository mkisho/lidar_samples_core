

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import math

# Importing the dataset
dataset = pd.read_csv('/home/mathias/catkin_ws/src/lidar_samples_reloaded/datasets/approachRV.csv')
X0 = dataset.iloc[:, 0].values
X1 = dataset.iloc[:, 1].values
X2 = dataset.iloc[:, 2].values

#for i in range(len(X2)):
#	X2[i]=X2[i].replace('[', '')
#	X2[i]=X2[i].replace(']', '')
#	X2[i]=float(X2[i])
print(X1)
print(X2.flatten())
fig, ax = plt.subplots()
ax.plot(X0,X1)
ax.plot(X0,X2)

ax.set(xlabel='time (s)', ylabel='Distance (Meters)',
       title='Distance predicted vs real')

plt.plot()
plt.show()


