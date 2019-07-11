# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 23:01:49 2019

@author: user1
"""
import pandas as pd
import os

year = '1999'

file = os.path.join(year,'/data_preprocessed_numerical.csv')
                    
df = pd.read_csv(file,sep=',')