import os
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn import preprocessing
from adalinesgd import AdalineSGD

clear = lambda: os.system('cls')
clear()

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

# Create scaler
scaler = preprocessing.StandardScaler()
# Transform the feature
X = scaler.fit_transform(X)

def plot_decision_regions(X, y, classifier, resolution=0.02, plot=plt):
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
	plot.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
	plot.axis(xmin=xx1.min(),xmax=xx1.max(), ymin=xx2.min(), ymax=xx2.max())

	# plot class examples
	for idx, cl in enumerate(np.unique(y)):
		plot.scatter(x=X[y == cl, 0],
					y=X[y == cl, 1],
					alpha=0.8,
					c=colors[idx],
					marker=markers[idx],
					label=cl,
					edgecolor='black')

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
ada_sgd = AdalineSGD(n_iter=15, eta=0.01, random_state=1).fit(X, y)

ax[0].plot(range(1, len(ada_sgd.cost_) + 1), ada_sgd.cost_, marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Average Cost')

plot_decision_regions(X, y, classifier=ada_sgd, plot=ax[1])
ax[1].set_title('Adaline - Stochastic Gradient Descent')
ax[1].set_xlabel('sepal length [standardized]')
ax[1].set_ylabel('petal length [standardized]')
ax[1].legend(loc='upper left')

plt.tight_layout()
plt.show()