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

x=[74.1, 74.77, 73.94, 73.61, 73.4]
plt.scatter(x,[0]*len(x),c="b")
plt.scatter(np.mean(x),0,c="r")
plt.show()