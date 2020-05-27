import numpy as np
import os
import matplotlib.pyplot as plt
#from scipy import stats
import scipy.stats as stats

clear = lambda: os.system('cls')
clear()

ages = np.random.randint(18,90,500)
print(stats.mode(ages))