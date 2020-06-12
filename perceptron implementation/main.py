import os
import pandas as pd
from sklearn import datasets

clear = lambda: os.system('cls')
clear()

s = os.path.join('https://archive.ics.uci.edu', 'ml','machine-learning-databases','iris','iris.data')

iris = datasets.load_iris()

df = pd.DataFrame(
	iris['data'], columns=iris['feature_names']
).assign(Species=iris['target_names'][iris['target']])

print(df)