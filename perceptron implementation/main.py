import os
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from perceptron import Perceptron
from matplotlib.colors import ListedColormap

clear = lambda: os.system('cls')
clear()

s = os.path.join('https://archive.ics.uci.edu', 'ml','machine-learning-databases','iris','iris.data')

#load iris dataset
iris = datasets.load_iris()

#convert to dataframe
df = pd.DataFrame(
	iris['data'], columns=iris['feature_names']
).assign(Species=iris['target_names'][iris['target']])

#print(df.head())

# select setosa and versicolor
y = df.iloc[0:100, 4].replace(["setosa", "versicolor"],[-1,1]).values

# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values

"""plt.scatter(X[:50, 0], X[:50, 1],color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel(df.columns[0])
plt.ylabel(df.columns[2])
plt.legend(loc="upper left")
plt.show()"""

"""ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()"""

def plot_decision_regions(X, y, classifier, resolution=0.02):
	# setup marker generator and color map
	markers = ('s', 'x', 'o', '^', 'v')
	colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
	cmap = ListedColormap(colors[:len(np.unique(y))])

	# plot the decision surface
	x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
	Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
	Z = Z.reshape(xx1.shape)
	plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
	plt.xlim(xx1.min(), xx1.max())
	plt.ylim(xx2.min(), xx2.max())

	# plot class examples
	for idx, cl in enumerate(np.unique(y)):
		plt.scatter(x=X[y == cl, 0],
					y=X[y == cl, 1],
					alpha=0.8,
					c=colors[idx],
					marker=markers[idx],
					label=cl,
					edgecolor='black')

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()