
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


labels = ('Horizontal',  'Vertical')
#, 'Approach')

svm_h = [3.0791211605072023, 0.27012526988983153]

rn_h = [19.817061376571656, 19.985271453857423]

rn= [100.0, 96.19847939175669, 98.91956782713085, 98.71948779511804, 99.11964785914365, 98.95958383353342]
svm=[100.0, 99.15966386554622, 99.75990396158463, 99.75990396158463, 99.83993597438976, 99.4797919167667]




x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots()
#rects1 = ax.bar(x, mae, width, label='Mean absolute error in angle prediction', align='center')
rects1 = ax.bar(x - width/2, svm_h, width, label='SVM')
rects2 = ax.bar(x + width/2, rn_h, width, label='NN')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Time (ms)')
ax.set_title('Time performance for the SVM and NN algorithms')
ax.set_xticks(x)
ax.set_xticklabels(labels,fontsize=8)
ax.legend()

#ax.bar_label(rects1, padding=3)
#ax.bar_label(rects2, padding=3)
ax.set_ylim(bottom=0,top=25)

fig.tight_layout()

plt.legend(['SVM','NN'], loc='lower right')

plt.show()
