
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


labels = ('Only houses',  'Many \ntrees', 'Only \ntower', 'Tower and  \ntrees', 'Tower and \nhouses','Tower and \npower plant')
#, 'Approach')

svm_h = [98.79951980792316, 93.87755102040816, 100.0, 96.71868747498999, 97.47899159663865, 99.15966386554622]
#, 100.0]
rn_h = [84.593837535014, 88.39535814325731, 98.51940776310524, 82.11284513805522, 91.55662264905963, 97.95918367346938]
#, 100.0]

rn= [100.0, 96.19847939175669, 98.91956782713085, 98.71948779511804, 99.11964785914365, 98.95958383353342]
#, 94.11389236545682]
svm=[100.0, 99.15966386554622, 99.75990396158463, 99.75990396158463, 99.83993597438976, 99.4797919167667]
#, 97.24655819774718]



x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
#rects1 = ax.bar(x, mae, width, label='Mean absolute error in angle prediction', align='center')
rects1 = ax.bar(x - width/2, svm_h, width, label='SVM')
rects2 = ax.bar(x + width/2, rn_h, width, label='NN')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy (%)')
ax.set_title('Performance in different scenarios')
ax.set_xticks(x)
ax.set_xticklabels(labels,fontsize=8)
ax.legend()

#ax.bar_label(rects1, padding=3)
#ax.bar_label(rects2, padding=3)
ax.set_ylim(bottom=80,top=100)

fig.tight_layout()

plt.legend(['SVM','NN'], loc='lower left')

plt.show()
