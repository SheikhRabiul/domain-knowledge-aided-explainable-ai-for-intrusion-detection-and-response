# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 11:30:28 2019

@author: Sheikh Rabiul Islam
Purpose: merging data in one place.
"""
#modules
import pandas as pd

#read raw data files
df_1 = pd.read_csv("dataset/MachineLearningCSV/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", sep =",")

df_2 = pd.read_csv("dataset/MachineLearningCSV/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv", sep =",")

df_3 = pd.read_csv("dataset/MachineLearningCSV/Friday-WorkingHours-Morning.pcap_ISCX.csv", sep =",")

df_4 = pd.read_csv("dataset/MachineLearningCSV/Monday-WorkingHours.pcap_ISCX.csv", sep =",")

df_5 = pd.read_csv("dataset/MachineLearningCSV/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv", sep =",")

df_6 = pd.read_csv("dataset/MachineLearningCSV/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", sep =",")

df_7 = pd.read_csv("dataset/MachineLearningCSV/Tuesday-WorkingHours.pcap_ISCX.csv", sep =",")

df_8 = pd.read_csv("dataset/MachineLearningCSV/Wednesday-workingHours.pcap_ISCX.csv", sep =",")

df = pd.concat([df_1, df_2, df_3, df_4, df_5, df_6, df_7, df_8], axis=0)   # 2,830,743 rows

#delete unnecessary dataframe to free some memory
del df_1
del df_2
del df_3
del df_4
del df_5
del df_6
del df_7
del df_8

# last column of the dataframe
df_last_l= list(df[df.columns[-1]])

#lable data int to two class from multiclass.
for i in range(len(df_last_l)):
    if df_last_l[i] == 'BENIGN':
        df_last_l[i] = 0
    else:
        df_last_l[i] = 1

# name the target column as Class        
df['Class'] = df_last_l

#drop the multi class clumn         
df = df.drop(df.columns[-2], axis =1)

#remove beginning and trailing spaces from column name
df.columns = df.columns.str.strip()

#write the merged dataframe to csv file, rename the index column as it is not sorted anymore. 
df.to_csv("dataset/MachineLearningCSV/combined.csv", sep = ",", index_label = "index_old") 

del df
        