# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 01:10:43 2019

@author: user1
"""

import pandas as pd

df =  pd.DataFrame({'A': ['a', 'b', 'a'], 'B': ['b', 'a', 'c'],'C': [1, 2, 3]})

df2 = pd.get_dummies(df)