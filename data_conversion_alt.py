# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:36:00 2019

@author: Sheikh Rabiul Islam
Purpose: numerical train, test set for few algorithms like GA; they don't require the dummy column version of data beforhand. 
"""

#import modules
import pandas as pd   
import numpy as np
import sqlite3
import time

start = time.time()

conn = sqlite3.connect("database/credit.sqlite")
curr = conn.cursor()

file_name1 = "data/data_preprocessed_numerical.csv"
file_name2 = "data/data_preprocessed_numerical_train_ids.csv"
file_name3 = "data/data_preprocessed_numerical_test_ids.csv"

df = pd.read_csv(file_name1,sep=",")
df_train = pd.read_csv(file_name2,sep=",",header=None, names = ['id'])
df_test = pd.read_csv(file_name3,sep=",",header=None, names = ['id'])

df.to_sql("data_numerical", conn, if_exists = "replace")
df_train.to_sql("data_numerical_train_ids", conn, if_exists = "replace", index=False, index_label='id')
df_test.to_sql("data_numerical_test_ids", conn, if_exists = "replace", index=False,index_label='id')
conn.commit()


df_train = pd.read_sql_query("select a.* from data_numerical a, data_numerical_train_ids b where a.`Unnamed: 0` = b.id;", conn)
df_train.drop(labels=['Unnamed: 0','index'],axis=1, inplace = True)
df_train.to_csv("data/data_preprocessed_numerical_train.csv",encoding='utf-8')

df_test = pd.read_sql_query("select a.* from data_numerical a, data_numerical_test_ids b where a.`Unnamed: 0` = b.id;", conn)
df_test.drop(labels=['Unnamed: 0','index'],axis=1, inplace = True)
df_test.to_csv("data/data_preprocessed_numerical_test.csv",encoding='utf-8')


X_train = df_train.iloc[:,0:-1].values
y_train = df_train.iloc[:,-1].values


################oversampling the minority class of training set #########

from imblearn.over_sampling import SMOTE 
# help available here: #https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)


df_dump_part1 = pd.DataFrame(X_train_res, columns=df_train.iloc[:,0:-1].columns.values)
df_dump_part2 = pd.DataFrame(y_train_res, columns=['defaulted'])   
df_dump = pd.concat([df_dump_part1,df_dump_part2], axis = 1)

df_dump.to_csv("data/data_preprocessed_numerical_train_res.csv",encoding='utf-8')

curr.close()
conn.close()

end = time.time()
print("checkpoint 1:", end-start)

