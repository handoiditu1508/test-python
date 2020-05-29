import os
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from sklearn.metrics import r2_score

#clear console
clear = lambda: os.system('cls')
clear()

#seed random function
np.random.seed(2)

#create random data
pageSpeeds = np.random.normal(3.0, 1.0, 100)
purchaseAmounts = np.random.normal(50.0, 30.0, 100)/pageSpeeds

#split pageSpeeds into training set and test set
trainX = pageSpeeds[:80]
testX = pageSpeeds[80:]
#split purchaseAmounts into training set and test set
trainY = purchaseAmounts[:80]
testY = purchaseAmounts[80:]

#calculate polynomial function
polynomial = np.poly1d(np.polyfit(trainX, trainY, 6))

#r squared of polynomial compare to test set
r2 = r2_score(testY, polynomial(testX))
print("r squared: "+str(r2))

xp = linspace(0, 7, 100)#[0, 0.07070707, ..., 7].count == 100
axes = plt.axes()
axes.set_xlim([0, 7])
axes.set_ylim([0, 200])
plt.scatter(trainX, trainY, c="g")
plt.scatter(testX, testY, c="b")
plt.plot(xp, polynomial(xp), c="r")
plt.show()