# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 21:44:42 2019

@author: user1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X_train = np.load('data/data_fully_processed_X_train_domain_features.npy')
y_train = np.load('data/data_fully_processed_y_train_domain_features.npy')
X_test = np.load('data/data_fully_processed_X_test_domain_features.npy')
y_test = np.load('data/data_fully_processed_y_test_domain_features.npy')


X = np.concatenate((X_train, X_test), axis=0)
#X_columns = ['Destination Port','Flow Duration','Total Fwd Packets','Total Backward Packets','Total Length of Fwd Packets','Total Length of Bwd Packets','Fwd Packet Length Max','Fwd Packet Length Min','Fwd Packet Length Mean','Fwd Packet Length Std','Bwd Packet Length Max','Bwd Packet Length Min','Bwd Packet Length Mean','Bwd Packet Length Std','Flow Bytes/s','Flow Packets/s','Flow IAT Mean','Flow IAT Std','Flow IAT Max','Flow IAT Min','Fwd IAT Total','Fwd IAT Mean','Fwd IAT Std','Fwd IAT Max','Fwd IAT Min','Bwd IAT Total','Bwd IAT Mean','Bwd IAT Std','Bwd IAT Max','Bwd IAT Min','Fwd PSH Flags','Bwd PSH Flags','Fwd URG Flags','Bwd URG Flags','Fwd Header Length','Bwd Header Length','Fwd Packets/s','Bwd Packets/s','Min Packet Length','Max Packet Length','Packet Length Mean','Packet Length Std','Packet Length Variance','FIN Flag Count','SYN Flag Count','RST Flag Count','PSH Flag Count','ACK Flag Count','URG Flag Count','CWE Flag Count','ECE Flag Count','Down/Up Ratio','Average Packet Size','Avg Fwd Segment Size','Avg Bwd Segment Size','Fwd Header Length.1','Fwd Avg Bytes/Bulk','Fwd Avg Packets/Bulk','Fwd Avg Bulk Rate','Bwd Avg Bytes/Bulk','Bwd Avg Packets/Bulk','Bwd Avg Bulk Rate','Subflow Fwd Packets','Subflow Fwd Bytes','Subflow Bwd Packets','Subflow Bwd Bytes','Init_Win_bytes_forward','Init_Win_bytes_backward','act_data_pkt_fwd','min_seg_size_forward','Active Mean','Active Std','Active Max','Active Min','Idle Mean','Idle Std','Idle Max','Idle Min']
X_columns = ['ACK Flag Count','Active Mean','Active Min','Average Packet Size','Bwd IAT Mean','Bwd Packet Length Min','Bwd Packet Length Std','Bwd Packets/s','Fwd IAT Mean','Fwd IAT Min','Fwd Packet Length Mean','Fwd Packets/s','Fwd PSH Flags','Flow Duration','Flow IAT Mean','Flow IAT Min','Flow IAT Std','Init_Win_bytes_forward','PSH Flag Count','Subflow Fwd Bytes','SYN Flag Count','Total Length of Fwd Packets']
y = np.concatenate((y_train, y_test), axis=0)
X_cloumns_d = { X_columns[i]:i for i in range(0, len(X_columns)) }

df_part1 = pd.DataFrame(X, columns=X_columns)
df_part2 = pd.DataFrame(y, columns=['Class'])   
data = pd.concat([df_part1,df_part2], axis = 1)    


#data = pd.read_csv('data/result_numeric.csv', index_col=0)
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