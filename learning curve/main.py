import os
from sklearn import datasets
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve

clear = lambda: os.system("cls")
clear()

#load dataset
df, y = datasets.load_breast_cancer(return_X_y=True, as_frame=True)

X=df.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

pipe_lr = make_pipeline(StandardScaler(),
						#PCA(n_components=2),
						LogisticRegression(penalty="l2",
						random_state=1,
						solver="lbfgs",
						max_iter=10000))

train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr,
														X=X_train,
														y=y_train,
														train_sizes=np.linspace(0.1, 1.0, 10),
														cv=10,
														n_jobs=1
)

#validation curve
"""
param_range = [0.001, 0.01, 0.1, 1, 1.0, 10.0, 100.0]
train_scores, test_scores = validation_curve(estimator=pipe_lr,
											X=X_train,
											y=y_train,
											param_name="logisticregression__C",
											param_range=param_range,
											cv=10
)
"""

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean,
		color="blue", marker="o",
		markersize=5, label="Training accuracy"
)

plt.fill_between(train_sizes,
				train_mean + train_std,
				train_mean - train_std,
				alpha = 0.15,
				color = "blue"
)

plt.plot(train_sizes, test_mean,
		color="green", linestyle="--",
		marker="s", markersize=5,
		label="Validation accuracy"
)

plt.fill_between(train_sizes,
				test_mean + test_std,
				test_mean - test_std,
				alpha = 0.15,
				color = "green"
)

#validation curve
"""
plt.plot(param_range, train_mean,
		color="blue", marker="o",
		markersize=5, label="Training accuracy"
)

plt.fill_between(param_range,
				train_mean + train_std,
				train_mean - train_std,
				alpha = 0.15,
				color = "blue"
)

plt.plot(param_range, test_mean,
		color="green", linestyle="--",
		marker="s", markersize=5,
		label="Validation accuracy"
)

plt.fill_between(param_range,
				test_mean + test_std,
				test_mean - test_std,
				alpha = 0.15,
				color = "green"
)
"""

plt.grid()
plt.xlabel("Number of training examples")
#validation curve
#plt.xlabel("Parameter C")
plt.ylabel("Accuracy")
plt.legend(loc = "lower right")
plt.ylim([0.8, 1.03])
plt.show()

"""
pipe_lr.fit(X_train, y_train)

y_predict = pipe_lr.predict(X_test)

print("test accuracy: %.3f" %pipe_lr.score(X_test, y_test))
"""