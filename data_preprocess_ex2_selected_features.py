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
attack_id = 12



# import data
dataset_train = pd.read_csv('data/data_preprocessed_numerical_train_all_features.csv', sep=',')
dataset_test = pd.read_csv('data/data_preprocessed_numerical_test_all_features.csv', sep=',')

# delete all records from training set containing the attack in attack_id
dataset_train = dataset_train[dataset_train['Class_all'] != 1]

#drop extra columns
dataset_train = dataset_train.drop(['Unnamed: 0', 'index', 'index_old', 'Class_all'], axis=1)
dataset_test = dataset_test.drop(['Unnamed: 0', 'index', 'index_old', 'Class_all'], axis=1)

X_train = dataset_train.iloc[:,0:-1].values
y_train = dataset_train.iloc[:,-1].values

X_test = dataset_test.iloc[:,0:-1].values
y_test = dataset_test.iloc[:,-1].values

end = time.time()
print("checkpoint 1:", end-start)

#dump onehot encoded training data
# save the fully processed data as binary for future use in any ML algorithm without any more preprocessing. 
np.save('data/data_fully_processed_X_train_selected_features.npy',X_train)
np.save('data/data_fully_processed_y_train_selected_features.npy',y_train)

print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

# save the fully processed data as binary for future use in any ML algorithm without any more preprocessing. 
np.save('data/data_fully_processed_X_test_selected_features.npy',X_test)
np.save('data/data_fully_processed_y_test_selected_features.npy',y_test)


end = time.time()
print("checkpoint 2:", end-start)

################oversampling the minority class of training set #########

from imblearn.over_sampling import SMOTE 
# help available here: #https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

# save the fully processed data as binary for future use in any ML algorithm without any more preprocessing. 
np.save('data/data_fully_processed_X_train_resampled_selected_features.npy',X_train_res)
np.save('data/data_fully_processed_y_train_resampled_selected_features.npy',y_train_res)

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))
print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))

