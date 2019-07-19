# Author: Sheikh Rabiul Islam
# Date: 07/13/2019; updated:  07/15/2019
# Purpose: load preprocessed data, delete records from a particular attack from the training set only. 
#		save the fully processed data as numpy array (binary: data/____.npy)

#import modules
import pandas as pd   
import numpy as np
import time
start = time.time()

# set attack id  (1-13) to delete from training set
config_file = 'config.txt'
config = pd.read_csv(config_file,sep=',', index_col =None)
attack_id = config.iloc[2,1]
print("attack_id: ", attack_id)

# import data
dataset_train = pd.read_csv('data/data_preprocessed_numerical_train_all_features.csv', sep=',')
dataset_test = pd.read_csv('data/data_preprocessed_numerical_test_all_features.csv', sep=',')

# delete all records from training set containing the attack in attack_id
dataset_train = dataset_train[dataset_train['Class_all'] != attack_id]

#drop extra columns
#old
#feature_selected = ['ACK Flag Count','Active Mean','Active Min','Average Packet Size','Bwd IAT Mean','Bwd Packet Length Min','Bwd Packet Length Std','Bwd Packets/s','Fwd IAT Mean','Fwd IAT Min','Fwd Packet Length Mean','Fwd Packets/s','Fwd PSH Flags','Flow Duration','Flow IAT Mean','Flow IAT Min','Flow IAT Std','Init_Win_bytes_forward','PSH Flag Count','Subflow Fwd Bytes','SYN Flag Count','Total Length of Fwd Packets', 'Class']

#now, excluding 'Bwd Packet Length Min' (benign)
#feature_selected = ['ACK Flag Count','Active Mean','Active Min','Average Packet Size','Bwd IAT Mean','Bwd Packet Length Std','Bwd Packets/s','Fwd IAT Mean','Fwd IAT Min','Fwd Packet Length Mean','Fwd Packets/s','Fwd PSH Flags','Flow Duration','Flow IAT Mean','Flow IAT Min','Flow IAT Std','Fwd IAT Min','Init_Win_bytes_forward','PSH Flag Count','Subflow Fwd Bytes','SYN Flag Count','Total Length of Fwd Packets', 'Class']

# overall top 3 added 
#feature_selected = ['Flow Bytes/s','Avg Fwd Segment Size','Packet Length Mean','ACK Flag Count','Active Mean','Active Min','Average Packet Size','Bwd IAT Mean','Bwd Packet Length Min','Bwd Packet Length Std','Bwd Packets/s','Fwd IAT Mean','Fwd IAT Min','Fwd Packet Length Mean','Fwd Packets/s','Fwd PSH Flags','Flow Duration','Flow IAT Mean','Flow IAT Min','Flow IAT Std','Init_Win_bytes_forward','PSH Flag Count','Subflow Fwd Bytes','SYN Flag Count','Total Length of Fwd Packets', 'Class']

#discarding less important
#feature_selected = ['Average Packet Size','Bwd IAT Mean','Bwd Packet Length Std','Bwd Packets/s','Fwd IAT Mean','Fwd IAT Min','Fwd Packet Length Mean','Flow Duration','Flow IAT Min','Flow IAT Std','Fwd IAT Min','Init_Win_bytes_forward','PSH Flag Count','Subflow Fwd Bytes','SYN Flag Count','Total Length of Fwd Packets', 'Class']
feature_selected = ['Bwd Packet Length Min','Average Packet Size','Bwd IAT Mean','Bwd Packet Length Std','Bwd Packets/s','Fwd IAT Mean','Fwd IAT Min','Fwd Packet Length Mean','Flow Duration','Flow IAT Min','Flow IAT Std','Fwd IAT Min','Init_Win_bytes_forward','Subflow Fwd Bytes','Total Length of Fwd Packets', 'Class']

#discarding less important; adding overall top 3
#feature_selected = ['Flow Bytes/s','Avg Fwd Segment Size','Packet Length Mean','Average Packet Size','Bwd IAT Mean','Bwd Packet Length Std','Bwd Packets/s','Fwd IAT Mean','Fwd IAT Min','Fwd Packet Length Mean','Flow Duration','Flow IAT Min','Flow IAT Std','Fwd IAT Min','Init_Win_bytes_forward','PSH Flag Count','Subflow Fwd Bytes','SYN Flag Count','Total Length of Fwd Packets', 'Class']


dataset_train = dataset_train[feature_selected]
dataset_test = dataset_test[feature_selected]


#dataset_train = dataset_train.drop(['Unnamed: 0', 'index', 'index_old', 'Class_all'], axis=1)
#dataset_test = dataset_test.drop(['Unnamed: 0', 'index', 'index_old', 'Class_all'], axis=1)

X_train = dataset_train.iloc[:,0:-1].values
y_train = dataset_train.iloc[:,-1].values

X_test = dataset_test.iloc[:,0:-1].values
y_test = dataset_test.iloc[:,-1].values

end = time.time()
print("checkpoint 1:", end-start)

#dump onehot encoded training data
# save the fully processed data as binary for future use in any ML algorithm without any more preprocessing. 
np.save('data/data_fully_processed_X_train_domain_features.npy',X_train)
np.save('data/data_fully_processed_y_train_domain_features.npy',y_train)

print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

# save the fully processed data as binary for future use in any ML algorithm without any more preprocessing. 
np.save('data/data_fully_processed_X_test_domain_features.npy',X_test)
np.save('data/data_fully_processed_y_test_domain_features.npy',y_test)


end = time.time()
print("checkpoint 2:", end-start)

################oversampling the minority class of training set #########

from imblearn.over_sampling import SMOTE 
# help available here: #https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

# save the fully processed data as binary for future use in any ML algorithm without any more preprocessing. 
np.save('data/data_fully_processed_X_train_resampled_domain_features.npy',X_train_res)
np.save('data/data_fully_processed_y_train_resampled_domain_features.npy',y_train_res)

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))
print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))

