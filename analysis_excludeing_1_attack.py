# Author: Sheikh Rabiul Islam
# Date: 07/13/2019; updated:  07/13/2019
# Purpose: analysis result of exclusing one attack's records from training set, test set remains intact.

#import modules
import pandas as pd   
import numpy as np
import time

config_file = 'config.txt'
config = pd.read_csv(config_file,sep=',', index_col =None)
attack_id = config.iloc[2,1]

filename = 'data/data_preprocessed_numerical_test_all_features.csv'
    
df_test = pd.read_csv(filename,sep=',')
df_y_pred = pd.read_csv('data/y_pred.csv',sep=',')

df = pd.concat([df_test,df_y_pred], axis = 1)     
df = df.drop(['Unnamed: 0'], axis =1)

df2= df.query('Class_all == @attack_id and Class == 1')

df3= df2.query('Class_all == @attack_id and Class == 1 and y_pred == 1')
      
print("actual # of attack ( attack ", attack_id, "): ", len(df2))
print("correctly detected (attack ", attack_id, "): ", len(df3))

if len(df3)!= 0:
    print("Percentage: ", (len(df3)/len(df2))*100)
else:
    print("Percentage: ", 0)