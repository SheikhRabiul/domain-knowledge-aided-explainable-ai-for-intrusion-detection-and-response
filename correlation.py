# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 21:44:42 2019

@author: user1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('data/result_numeric.csv', index_col=0)
corr = data.corr()
corr.to_csv("data/correlation_matrix.csv", sep=',')
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(data.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(data.columns)
ax.set_yticklabels(data.columns)
plt.show()