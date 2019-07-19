# Author: Sheikh Rabiul Islam
# Date: 07/15/2019
# Purpose: feature selection

#import modules
import pandas as pd   
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor

output_folder ='data'

X_train = np.load('data/data_fully_processed_X_train_domain_features.npy')
y_train = np.load('data/data_fully_processed_y_train_domain_features.npy')
X_test = np.load('data/data_fully_processed_X_test_domain_features.npy')
y_test = np.load('data/data_fully_processed_y_test_domain_features.npy')


X = np.concatenate((X_train, X_test), axis=0)
#X_columns = ['Destination Port','Flow Duration','Total Fwd Packets','Total Backward Packets','Total Length of Fwd Packets','Total Length of Bwd Packets','Fwd Packet Length Max','Fwd Packet Length Min','Fwd Packet Length Mean','Fwd Packet Length Std','Bwd Packet Length Max','Bwd Packet Length Min','Bwd Packet Length Mean','Bwd Packet Length Std','Flow Bytes/s','Flow Packets/s','Flow IAT Mean','Flow IAT Std','Flow IAT Max','Flow IAT Min','Fwd IAT Total','Fwd IAT Mean','Fwd IAT Std','Fwd IAT Max','Fwd IAT Min','Bwd IAT Total','Bwd IAT Mean','Bwd IAT Std','Bwd IAT Max','Bwd IAT Min','Fwd PSH Flags','Bwd PSH Flags','Fwd URG Flags','Bwd URG Flags','Fwd Header Length','Bwd Header Length','Fwd Packets/s','Bwd Packets/s','Min Packet Length','Max Packet Length','Packet Length Mean','Packet Length Std','Packet Length Variance','FIN Flag Count','SYN Flag Count','RST Flag Count','PSH Flag Count','ACK Flag Count','URG Flag Count','CWE Flag Count','ECE Flag Count','Down/Up Ratio','Average Packet Size','Avg Fwd Segment Size','Avg Bwd Segment Size','Fwd Header Length.1','Fwd Avg Bytes/Bulk','Fwd Avg Packets/Bulk','Fwd Avg Bulk Rate','Bwd Avg Bytes/Bulk','Bwd Avg Packets/Bulk','Bwd Avg Bulk Rate','Subflow Fwd Packets','Subflow Fwd Bytes','Subflow Bwd Packets','Subflow Bwd Bytes','Init_Win_bytes_forward','Init_Win_bytes_backward','act_data_pkt_fwd','min_seg_size_forward','Active Mean','Active Std','Active Max','Active Min','Idle Mean','Idle Std','Idle Max','Idle Min']
X_columns =['ACK Flag Count','Active Mean','Active Min','Average Packet Size','Bwd IAT Mean','Bwd Packet Length Min','Bwd Packet Length Std','Bwd Packets/s','Fwd IAT Mean','Fwd IAT Min','Fwd Packet Length Mean','Fwd Packets/s','Fwd PSH Flags','Flow Duration','Flow IAT Mean','Flow IAT Min','Flow IAT Std','Init_Win_bytes_forward','PSH Flag Count','Subflow Fwd Bytes','SYN Flag Count','Total Length of Fwd Packets']
y = np.concatenate((y_train, y_test), axis=0)

X_cloumns_d = { X_columns[i]:i for i in range(0, len(X_columns)) }

for i in range(0,len(X_columns)):
    print(i,',',X_columns[i])
    

#configure here
threshold_feature_score = 0


def scale_a_number(inpt, to_min, to_max, from_min, from_max):
    return (to_max-to_min)*(inpt-from_min)/(from_max-from_min)+to_min

def scale_a_list(l, to_min, to_max):
    return [scale_a_number(i, to_min, to_max, min(l), max(l)) for i in l]


############ Random Forest Regressor #################

forest = RandomForestRegressor(n_estimators=10, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=4, random_state=None, verbose=0, warm_start=False)   
forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
data= (sorted(zip(map(lambda x: round(x, 4), forest.feature_importances_), X_columns),reverse=True))
#converting the list to a datafreame
df_result_rfc = pd.DataFrame(data,columns=['score','feature'])
df_result_rfc.insert(1,"scaled_score", scale_a_list(df_result_rfc['score'],0,1))
print("top 5 features using Random forest regressor:")
print(df_result_rfc.head())
f_name = 'selected_features_with_rank_domain.csv'
f_path = os.path.join(output_folder, f_name)
print("saving  ranked features in ")
print(f_path)
df_result_rfc.to_csv(f_path, sep=',')
    
print("******************* ending feature selection ********************")


print("\n ******************* discarding  insignificant features (score 0) ********************")

df_result_rfc_new = df_result_rfc[df_result_rfc['scaled_score']>0]
df_result_rfc_new = df_result_rfc_new.drop(['score','scaled_score'], axis=1)
f_name = 'selected_features_domain.csv'
f_path = os.path.join(output_folder, f_name)
print("saving  filtered features features in ")
print(f_path)
df_result_rfc_new.to_csv(f_path, sep=',')
