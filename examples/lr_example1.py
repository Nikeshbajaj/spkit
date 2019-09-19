'''
Signal Processing Toolkit ->ML
Machine Learning from scrach
Logistic Regression - Example 1 

@Author _ Nikesh Bajaj
PhD Student at Queen Mary University of London 
Conact _ http://nikeshbajaj.in 
Github:: https://github.com/Nikeshbajaj
n[dot]bajaj@qmul.ac.uk
bajaj[dot]nikkey@gmail.com
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


from spkit.data import dataGen as ds
from spkit.ml import LR



np.random.seed(1)
plt.close('all')

dtype = ['MOONS','GAUSSIANS','LINEAR','SINUSOIDAL','SPIRAL']

X, y,_ = ds.create_dataset(200, dtype[3],0.05,varargin = 'PRESET')

#Normalizing
means = np.mean(X,0)
stds  = np.std(X,0)
X = (X-means)/stds

#In cureent version LR takes X and y as shape (nf,n) and (n,1)
X = X.T
y = y[None,:]

print(X.shape, y.shape)

clf = LR(X,y,alpha=0.0003,polyfit=True,degree=5,lambd=2)

plt.ion()
fig=plt.figure(figsize=(8,4))
gs=GridSpec(1,2)
ax1=fig.add_subplot(gs[0,0])
ax2=fig.add_subplot(gs[0,1])

for i in range(100):
    clf.fit(X,y,itr=10,verbose=True)
    ax1.cla()
    clf.Bplot(ax1,hardbound=False)
    ax2.cla()
    clf.LCurvePlot(ax2)
    fig.canvas.draw()
    plt.pause(0.001)