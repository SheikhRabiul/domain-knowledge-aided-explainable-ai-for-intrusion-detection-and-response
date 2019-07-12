# Author: Sheikh Rabiul Islam
# Date: 07/10/2019; updated:  07/10/2019
# Purpose: remove unimportant features; labelEncode; onHotEncode;Scale; resample minority class;
#		save the fully processed data as numpy array (binary: data/____.npy)

#import modules
import pandas as pd   
import numpy as np
import time
start = time.time()

#maximum finite value in any cell of the dataset. Infinity value in any cell is replaced with with this value. 
max_value = 655453030.0
# import data
dataset = pd.read_csv('data/combined_sampled.csv', sep=',')

# seperate the dependent (target) variaable
X = dataset.iloc[:,0:-1].values
X_columns = dataset.iloc[:,0:-1].columns.values
y = dataset.iloc[:,-1].values

del(dataset)
#X_bk = pd.DataFrame(data=X, columns =X_columns ) 

from sklearn.preprocessing import LabelEncoder
df_dump_part1 = pd.DataFrame(X, columns=X_columns)
df_dump_part2 = pd.DataFrame(y, columns=['Class'])   
df_dump = pd.concat([df_dump_part1,df_dump_part2], axis = 1)     
df_dump.to_csv("data/data_preprocessed_numerical_alt.csv",encoding='utf-8')   # keeping a backup of preprocessed numerical data.
end = time.time()
print("checkpoint 1:", end-start)

start = time.time()

# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

X = np.array(X, dtype=float) # this is required for checking infinite and null below
#X_bk = pd.DataFrame(data=X, columns =X_columns ) 
# replace infinite with max_value, null with 0.0
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        k = X[i,j]
        if not np.isfinite(k):
            X[i,j] = max_value
        if np.isnan(k):
            X[i,j] = 0.0

# Feature Scaling (scaling all attributes/featues in the same scale)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


#add index to X to indentify the rows after split.
index = np.arange(len(X)).reshape(len(X),1)
X = np.hstack((index,X))

#########seperating training and test set  ##################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=42, stratify=y)


# keeping a backup of preprocessed numerical data (train and test).
df_dump_train =  pd.DataFrame(X_train[:,0], columns=['id'])
df_dump_train.to_csv("data/data_preprocessed_numerical_train_ids_alt.csv",encoding='utf-8')
df_dump_test =  pd.DataFrame(X_test[:,0], columns=['id'])
df_dump_test.to_csv("data/data_preprocessed_numerical_test_ids_alt.csv",encoding='utf-8')

end = time.time()
print("checkpoint 2:", end-start)

"""df_dump_test = df_dump
for i in range(len(X_test)):
    df_dump_train.drop([X_test[i,0]], inplace = True)
df_dump_train.to_csv("data/data_preprocessed_numerical_train.csv",encoding='utf-8')   # keeping a backup of preprocessed numerical data (train).

for i in range(len(X_train)):
    df_dump_test.drop([X_train[i,0]], inplace = True)    
df_dump_test.to_csv("data/data_preprocessed_numerical_test.csv",encoding='utf-8')   # keeping a backup of preprocessed numerical data (test).
"""
# index in X is no more needed; drop it
start = time.time()

X_train = np.delete(X_train,0,1)
X_test = np.delete(X_test,0,1)

del(X)   # free some memory; encoded (onehot) data takes lot of memory
del(y) # free some memory; encoded (onehot) data takes lot of memory

#dump onehot encoded training data
# save the fully processed data as binary for future use in any ML algorithm without any more preprocessing. 
np.save('data/data_fully_processed_X_train_alt.npy',X_train)
np.save('data/data_fully_processed_y_train_alt.npy',y_train)

print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

# save the fully processed data as binary for future use in any ML algorithm without any more preprocessing. 
np.save('data/data_fully_processed_X_test_alt.npy',X_test)
np.save('data/data_fully_processed_y_test_alt.npy',y_test)


end = time.time()
print("checkpoint 3:", end-start)

################oversampling the minority class of training set #########

from imblearn.over_sampling import SMOTE 
# help available here: #https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

# save the fully processed data as binary for future use in any ML algorithm without any more preprocessing. 
np.save('data/data_fully_processed_X_train_resampled_alt.npy',X_train_res)
np.save('data/data_fully_processed_y_train_resampled_alt.npy',y_train_res)

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))
print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))

