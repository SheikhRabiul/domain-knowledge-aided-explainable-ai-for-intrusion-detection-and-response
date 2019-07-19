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
feature_selected = ['ACK Flag Count','Active Mean','Active Min','Average Packet Size','Bwd IAT Mean',
                    'Bwd Packet Length Std','Bwd Packets/s','Fwd IAT Mean','Fwd IAT Min','Fwd Packet Length Mean','Fwd Packets/s',
                    'Fwd PSH Flags','Flow Duration','Flow IAT Mean','Flow IAT Min','Flow IAT Std','Init_Win_bytes_forward',
                    'PSH Flag Count','Subflow Fwd Bytes','SYN Flag Count','Total Length of Fwd Packets', 'Class']

dataset_train = dataset_train[feature_selected]
dataset_test = dataset_test[feature_selected]
from sklearn import preprocessing

dataset_train_class = np.array(dataset_train['Class'])
xx = dataset_train.iloc[:,0:-1].values
xx_columns = dataset_train.columns
xx_columns = xx_columns[0:-1]
min_max_scaler = preprocessing.MinMaxScaler()
xx_scaled = min_max_scaler.fit_transform(xx)
dataset_train = pd.DataFrame(xx_scaled)
df_part1 = pd.DataFrame(xx_scaled, columns=xx_columns)
df_part2 = pd.DataFrame(dataset_train_class, columns=['Class'])   
dataset_train = pd.concat([df_part1,df_part2], axis = 1) 


dataset_train.insert(loc = 21, column = 'C', value = np.zeros(len(dataset_train)))
dataset_train.insert(loc = 22, column = 'I', value = np.zeros(len(dataset_train)))
dataset_train.insert(loc = 23, column = 'A', value = np.zeros(len(dataset_train)))

dataset_train['C'] = dataset_train['C'] + dataset_train['ACK Flag Count']

dataset_train['C'] = dataset_train['C'] + dataset_train['Active Min']*.5
dataset_train['A'] = dataset_train['A'] + dataset_train['Active Min']*.5

dataset_train['A'] = dataset_train['A'] + dataset_train['Average Packet Size']

dataset_train['A'] = dataset_train['A'] + dataset_train['Bwd IAT Mean']

dataset_train['A'] = dataset_train['A'] + dataset_train['Bwd Packet Length Std']*.5
dataset_train['C'] = dataset_train['C'] + dataset_train['Bwd Packet Length Std']*.5

dataset_train['C'] = dataset_train['C'] + dataset_train['Bwd Packets/s']*.333
dataset_train['I'] = dataset_train['I'] + dataset_train['Bwd Packets/s']*.333
dataset_train['A'] = dataset_train['A'] + dataset_train['Bwd Packets/s']*.333

dataset_train['A'] = dataset_train['A'] + dataset_train['Fwd IAT Mean']

dataset_train['A'] = dataset_train['A'] + dataset_train['Fwd IAT Min']

dataset_train['C'] = dataset_train['C'] + dataset_train['Fwd Packet Length Mean']*.333
dataset_train['I'] = dataset_train['I'] + dataset_train['Fwd Packet Length Mean']*.333
dataset_train['A'] = dataset_train['A'] + dataset_train['Fwd Packet Length Mean']*.333

dataset_train['C'] = dataset_train['C'] + dataset_train['Fwd Packets/s']

dataset_train['C'] = dataset_train['C'] + dataset_train['Fwd Packets/s']

dataset_train['C'] = dataset_train['C'] + dataset_train['Fwd PSH Flags']

dataset_train['A'] = dataset_train['A'] + dataset_train['Flow Duration']*.5
dataset_train['C'] = dataset_train['C'] + dataset_train['Flow Duration']*.5

dataset_train['A'] = dataset_train['A'] + dataset_train['Flow IAT Mean']

dataset_train['A'] = dataset_train['A'] + dataset_train['Flow IAT Min']

dataset_train['A'] = dataset_train['A'] + dataset_train['Flow IAT Std']

dataset_train['C'] = dataset_train['C'] + dataset_train['Init_Win_bytes_forward']*.333
dataset_train['I'] = dataset_train['I'] + dataset_train['Init_Win_bytes_forward']*.333
dataset_train['A'] = dataset_train['A'] + dataset_train['Init_Win_bytes_forward']*.333

dataset_train['C'] = dataset_train['C'] + dataset_train['PSH Flag Count']

dataset_train['C'] = dataset_train['C'] + dataset_train['Subflow Fwd Bytes']*.333
dataset_train['I'] = dataset_train['I'] + dataset_train['Subflow Fwd Bytes']*.333
dataset_train['A'] = dataset_train['A'] + dataset_train['Subflow Fwd Bytes']*.333

dataset_train['C'] = dataset_train['C'] + dataset_train['SYN Flag Count']

dataset_train['C'] = dataset_train['C'] + dataset_train['Total Length of Fwd Packets']*.333
dataset_train['I'] = dataset_train['I'] + dataset_train['Total Length of Fwd Packets']*.333
dataset_train['A'] = dataset_train['A'] + dataset_train['Total Length of Fwd Packets']*.333



dataset_test_class = np.array(dataset_test['Class'])
xx = dataset_test.iloc[:,0:-1].values
xx_columns = dataset_test.columns
xx_columns = xx_columns[0:-1]
min_max_scaler = preprocessing.MinMaxScaler()
xx_scaled = min_max_scaler.fit_transform(xx)
dataset_test = pd.DataFrame(xx_scaled)
df_part1 = pd.DataFrame(xx_scaled, columns=xx_columns)
df_part2 = pd.DataFrame(dataset_test_class, columns=['Class'])   
dataset_test = pd.concat([df_part1,df_part2], axis = 1) 



dataset_test.insert(loc = 21, column = 'C', value = np.zeros(len(dataset_test)))
dataset_test.insert(loc = 22, column = 'I', value = np.zeros(len(dataset_test)))
dataset_test.insert(loc = 23, column = 'A', value = np.zeros(len(dataset_test)))

dataset_test['C'] = dataset_test['C'] + dataset_test['ACK Flag Count']

dataset_test['C'] = dataset_test['C'] + dataset_test['Active Min']*.5
dataset_test['A'] = dataset_test['A'] + dataset_test['Active Min']*.5

dataset_test['A'] = dataset_test['A'] + dataset_test['Average Packet Size']

dataset_test['A'] = dataset_test['A'] + dataset_test['Bwd IAT Mean']

dataset_test['A'] = dataset_test['A'] + dataset_test['Bwd Packet Length Std']*.5
dataset_test['C'] = dataset_test['C'] + dataset_test['Bwd Packet Length Std']*.5

dataset_test['C'] = dataset_test['C'] + dataset_test['Bwd Packets/s']*.333
dataset_test['I'] = dataset_test['I'] + dataset_test['Bwd Packets/s']*.333
dataset_test['A'] = dataset_test['A'] + dataset_test['Bwd Packets/s']*.333

dataset_test['A'] = dataset_test['A'] + dataset_test['Fwd IAT Mean']

dataset_test['A'] = dataset_test['A'] + dataset_test['Fwd IAT Min']

dataset_test['C'] = dataset_test['C'] + dataset_test['Fwd Packet Length Mean']*.333
dataset_test['I'] = dataset_test['I'] + dataset_test['Fwd Packet Length Mean']*.333
dataset_test['A'] = dataset_test['A'] + dataset_test['Fwd Packet Length Mean']*.333

dataset_test['C'] = dataset_test['C'] + dataset_test['Fwd Packets/s']

dataset_test['C'] = dataset_test['C'] + dataset_test['Fwd Packets/s']

dataset_test['C'] = dataset_test['C'] + dataset_test['Fwd PSH Flags']

dataset_test['A'] = dataset_test['A'] + dataset_test['Flow Duration']*.5
dataset_test['C'] = dataset_test['C'] + dataset_test['Flow Duration']*.5

dataset_test['A'] = dataset_test['A'] + dataset_test['Flow IAT Mean']

dataset_test['A'] = dataset_test['A'] + dataset_test['Flow IAT Min']

dataset_test['A'] = dataset_test['A'] + dataset_test['Flow IAT Std']

dataset_test['C'] = dataset_test['C'] + dataset_test['Init_Win_bytes_forward']*.333
dataset_test['I'] = dataset_test['I'] + dataset_test['Init_Win_bytes_forward']*.333
dataset_test['A'] = dataset_test['A'] + dataset_test['Init_Win_bytes_forward']*.333

dataset_test['C'] = dataset_test['C'] + dataset_test['PSH Flag Count']

dataset_test['C'] = dataset_test['C'] + dataset_test['Subflow Fwd Bytes']*.333
dataset_test['I'] = dataset_test['I'] + dataset_test['Subflow Fwd Bytes']*.333
dataset_test['A'] = dataset_test['A'] + dataset_test['Subflow Fwd Bytes']*.333

dataset_test['C'] = dataset_test['C'] + dataset_test['SYN Flag Count']

dataset_test['C'] = dataset_test['C'] + dataset_test['Total Length of Fwd Packets']*.333
dataset_test['I'] = dataset_test['I'] + dataset_test['Total Length of Fwd Packets']*.333
dataset_test['A'] = dataset_test['A'] + dataset_test['Total Length of Fwd Packets']*.333


feature_selected = ['C','I','A','Class']
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

