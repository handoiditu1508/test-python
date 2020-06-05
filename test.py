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

x=[1, 1, 1, 3, 3, 4, 5, 6, 6, 8, 8, 8]
y=[7, 6, 5, 3, 2, 1, 1, 2, 3, 5, 6, 7]
plt.scatter(x, y, c="gray", marker="s")
plt.scatter(3, 7.5, c="red", )
plt.scatter(3, 4, c="green")
plt.scatter(7.5, 4, c="blue")
plt.show()