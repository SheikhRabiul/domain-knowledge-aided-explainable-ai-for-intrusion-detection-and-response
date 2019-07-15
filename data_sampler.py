# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 23:30:28 2019

@author: Sheikh Rabiul Islam
Purpose: sampling data.
"""
#modules
import pandas as pd

#read the combined data files
df = pd.read_csv("dataset/MachineLearningCSV/combined.csv", sep =",", dtype='unicode')

########################## sampling data [stratified] ######################################
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

columns = list(df.columns.values)

X_columns = columns[0:-1]

from sklearn.model_selection import train_test_split
X_remaining, X_sampled, y_remaining, y_sampled = train_test_split( X, y, test_size=500000, random_state=42, stratify=y)



df_dump_part1 = pd.DataFrame(X_sampled, columns=X_columns)
df_dump_part2 = pd.DataFrame(y_sampled, columns=['Class'])   
df_dump = pd.concat([df_dump_part1,df_dump_part2], axis = 1)     
df_dump.to_csv("data/combined_sampled.csv",encoding='utf-8')  


del df
del df_dump_part1
del df_dump_part2
del df_dump
        