import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
from scipy.stats import expon
from scipy.stats import binom
from pylab import *

clear = lambda: os.system('cls')
clear()

pageSpeeds = np.random.normal(3.0,1.0,100)
purchaseAmount = 100 - (pageSpeeds + np.random.normal(0,0.1,100))*3
scatter(pageSpeeds, purchaseAmount)
plt.plot(pageSpeeds, purchaseAmount,".")
plt.show()