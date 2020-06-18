import os
import numpy as np
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz

#clear console
clear = lambda: os.system('cls')
clear()

#read file to dataframe
input_file = "./data/play_goft.csv"
df = pd.read_csv(input_file, header = 0)

#convert data type to numeric
d = {"Sunny": 0, "Overcast": 1, "Rain": 2}
df["Outlook"] = df["Outlook"].map(d)
d = {"Weak": 0, "Strong": 1}
df["Wind"] = df["Wind"].map(d)
d = {"Yes": 0, "No": 1}
df["Decision"] = df["Decision"].map(d)

features = list(df.columns[:4])
print(features)

y = df["Decision"]
x = df[features]
clf = tree.DecisionTreeClassifier()
#clf = RandomForestClassifier(n_estimators = 10)
clf = clf.fit(x,y)

#try predictions
print(clf.predict([[1,100,50,0],[2,25,80,1]]))

#display tree
#tree.plot_tree(clf)
#plt.show()

dot_data = export_graphviz(clf,
						   filled=True,
						   rounded=True,
						   class_names=["Yes", "No"],
						   feature_names=df.columns[:len(df.columns)-1],
						   out_file=None)
graph = graph_from_dot_data(dot_data)
graph.write_png('tree.png')