import os
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

clear = lambda: os.system('cls')
clear()

def createClusteredData(n, k):
    np.random.seed(10)
    pointsPerCluster = float(n)/k
    x=[]
    for i in range(k):
        incomeCentroid = np.random.uniform(20000.0, 200000.0)
        ageCentroid = np.random.uniform(20.0, 70.0)
        for j in range(int(pointsPerCluster)):
            x.append([np.random.normal(incomeCentroid, 10000.0), np.random.normal(ageCentroid, 2.0)])
    x = np.array(x)
    return x

#create test data
data = createClusteredData(100, 5)

#choosing K
model = KMeans(n_clusters = 5)

#fit data to KMeans object
#scale data to normalize it, center data around 0
#important for good result
model = model.fit(scale(data))

print(model.labels_)

plt.figure(figsize = (8, 6))
plt.scatter(data[:,0], data[:,1], c=model.labels_.astype(float))
plt.show()