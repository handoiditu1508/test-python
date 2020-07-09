import os
from sklearn import datasets
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import learning_curve

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

#train_sizes, train_scores, test_scores = learning_curve(estimator-)

pipe_lr.fit(X_train, y_train)

y_predict = pipe_lr.predict(X_test)

print("test accuracy: %.3f" %pipe_lr.score(X_test, y_test))