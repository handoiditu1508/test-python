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

a = np.array([[1,2,3],[3,4,5],[5,6,7]])
b = np.array([[11,12,13],[13,14,15],[15,16,17]])
print(a[:2,:2])
print(b[0,:2])
print(np.dot(a[:2,:2],b[0,:2]))