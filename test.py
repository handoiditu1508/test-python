import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
from scipy.stats import expon
from scipy.stats import binom
from pylab import *
from sklearn.metrics import r2_score
import pandas as pd
import statsmodels.api as sm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

clear = lambda: os.system('cls')
clear()

np.random.seed(2)
a = []
for i in range(5):
	a.append([int(np.random.normal(500, 500)), int(np.random.normal(500, 500))])
a=array(a)
b = scale(a)
plt.scatter(a[:,0],a[:,1],c="r")
#plt.scatter(b[:,0],b[:,1],c="b")
plt.show()