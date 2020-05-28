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

clear = lambda: os.system('cls')
clear()

df = pd.read_excel("http://cdn.sundog-soft.com/Udemy/DataScience/cars.xls")
df["Model_ord"] = pd.Categorical(df.Model).codes
x=df[["Mileage","Model_ord","Doors"]]
y=df[["Price"]]

X1=sm.add_constant(x)
est=sm.OLS(y,X1).fit()

#print(est.summary())
print(y.groupby(df.Doors).mean())