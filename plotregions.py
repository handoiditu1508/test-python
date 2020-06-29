def plot_decision_regions(X, y, classifier, resolution=0.02, plot=plt):
	# setup marker generator and color map
	markers = ('s', 'x', 'o', '^', '1', 'p', 'P', '*', 'h', 'X', 'd')
	colors = ('red', 'blue', 'green', 'yellow', 'purple', 'cyan', 'pink', 'gray', 'brown', 'gold', 'silver')
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