import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
from scipy.stats import expon
from scipy.stats import binom

clear = lambda: os.system('cls')
clear()

data = np.ones(100)
data[70:] -= np.arange(30)
print(data)