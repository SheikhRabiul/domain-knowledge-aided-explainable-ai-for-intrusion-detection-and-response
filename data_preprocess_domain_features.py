# Author: Sheikh Rabiul Islam
# Date: 07/13/2019; updated:  07/13/2019
# Purpose: load preprocessed data, drop features which are not related to domain.
#		save the fully processed data as numpy array (binary: data/____.npy)

#import modules
import pandas as pd   
import numpy as np
import time
start = time.time()

# import previouly processed data with all features;
dataset_train = pd.read_csv('data/data_preprocessed_numerical_train_all_features.csv', sep=',')
dataset_test = pd.read_csv('data/data_preprocessed_numerical_test_all_features.csv', sep=',')

# delete all records from training set containing the attack in attack_id
#dataset_train = dataset_train[dataset_train['Class_all'] != attack_id]

feature_selected = ['ACK Flag Count','Active Mean','Active Min','Average Packet Size','Bwd IAT Mean','Bwd Packet Length Min','Bwd Packet Length Std','Bwd Packets/s','Fwd IAT Mean','Fwd IAT Min','Fwd Packet Length Mean','Fwd Packets/s','Fwd PSH Flags','Flow Duration','Flow IAT Mean','Flow IAT Min','Flow IAT Std','Fwd IAT Min','Init_Win_bytes_forward','PSH Flag Count','Subflow Fwd Bytes','SYN Flag Count','Total Length of Fwd Packets', 'Class']

dataset_train = dataset_train[feature_selected]
dataset_test = dataset_test[feature_selected]

#renaming features
#features_new_names = ['index_old','ACK Flag Count - C','Active Mean - AC','Active Min - A','Avg Packet Size - A','Bwd IAT Mean - A','Bwd Packet Length Min','Bwd Packet Length Std - AC','Bwd Packets/s - CIA','Fwd IAT Mean - A','Fwd IAT Min - A','Fwd Packet Length Mean - CIA','Fwd Packets/s - C','Fwd PSH Flags - C','Flow Duration - AC','Flow IAT Mean - A','Flow IAT Min - A','Flow IAT Std - A','Fwd IAT Min - A','Init Win Bytes Fwd - CIA','PSH Flag Count - C','Subflow Fwd Bytes - CIA','SYN Flag Count - C','Total Length of Fwd Packets - CIA', 'class_all','Class']
#dataset.columns = features_new_names

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

