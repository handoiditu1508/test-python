import os
from sklearn import datasets
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from scipy import interp

clear = lambda: os.system("cls")
clear()

#load dataset
df, y = datasets.load_breast_cancer(return_X_y=True, as_frame=True)

X=df.values
y=np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

pipe_lr = make_pipeline(
	StandardScaler(),
	PCA(n_components=2),
	LogisticRegression(
		penalty="l2",
		random_state=1,
		solver="lbfgs",
		C=100.0
	)
)

X_train2 = X_train[:,[4,14]]

cv = list(StratifiedKFold(n_splits=3, random_state=1, shuffle=True).split(X_train, y_train))

fig = plt.figure(figsize=(7, 5))

mean_tpr = 0.0#mean true positive rate
mean_fpr = np.linspace(0, 1, 100)#mean false positive rate
all_tpr = []

for i, (train, test) in enumerate(cv):
	probas = pipe_lr.fit(
		X_train2[train],
		y_train[train]
	).predict_proba(X_train2[test])
	fpr, tpr, thresholds = roc_curve(
		y_train[test],
		probas[:, 1],
		pos_label=1
	)
	print(test.shape)
	print(fpr.shape)
	print(tpr.shape)
	print(thresholds.shape)
	temp = input("press Enter to continue")