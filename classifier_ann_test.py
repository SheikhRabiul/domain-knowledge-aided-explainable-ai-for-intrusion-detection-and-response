# Author: Sheikh Rabiul Islam
#template from Deep Learning A-Z, udemy.com
# Date: 02/22/2019
# Purpose: ANN

#import libraries
import pandas as pd   
import numpy as np
import time
#configurations
config_file = 'config.txt'
config = pd.read_csv(config_file,sep=',', index_col =None)
resample_data =  1#config.iloc[0,1] #0 or 1

print("Neural Network:",resample_data)
start = time.time()


#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state =0)

#print('Train: %s | test: %s' % (train_indices, test_indices))      
X_train = np.load('data/data_fully_processed_X_train.npy')
y_train = np.load('data/data_fully_processed_y_train.npy')

if resample_data == 1:
    X_train = np.load('data/data_fully_processed_X_train_resampled.npy')
    y_train = np.load('data/data_fully_processed_y_train_resampled.npy')

#X_train = np.load('data/result_numeric_one_hot_encoded_X_train.npy')
#y_train = np.load('data/result_numeric_one_hot_encoded_y_train.npy')


X_test = np.load('data/data_fully_processed_X_test.npy')
y_test = np.load('data/data_fully_processed_y_test.npy')

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout



input_dim = X_train.shape[1]
units = int(input_dim/2)+1
"""
# Initialising the ANN
classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(units = units, kernel_initializer = 'uniform', activation = 'relu', input_dim = input_dim))
classifier.add(Dropout(rate = 0.1))
# Adding the second hidden layer
classifier.add(Dense(units = units, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(rate = 0.1))
# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 1000, epochs = 20) #, class_weight = {0:1.0,1:100.0})
# Part 3 - Making predictions and evaluating the model
# Predicting the Test set results
y_pred_probab = classifier.predict(X_test)

y_pred = np.zeros(len(y_test))
for i in range(len(y_test)):
    if y_pred_probab[i] > .5:
        y_pred[i] =1
        

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Part 4 - Evaluating, Improving and Tuning the ANN
"""
"""
# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = units, kernel_initializer = 'uniform', activation = 'relu', input_dim = input_dim))
    classifier.add(Dense(units = units, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10000, epochs = 5)

classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred_probab = y_pred
y_pred = (y_pred_probab > 0.5)

#accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
#mean = accuracies.mean()
#variance = accuracies.std()

# Improving the ANN
# Dropout Regularization to reduce overfitting if needed


"""

# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = units, kernel_initializer = 'uniform', activation = 'relu', input_dim = input_dim))
    classifier.add(Dense(units = units, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
#parameters = {'batch_size': [25, 32],
#              'epochs': [10, 20],
#              'optimizer': ['adam', 'rmsprop']}
parameters = {'batch_size': [100000],
              'epochs': [5],
              'optimizer': ['adam']
              }
#parameters = {'batch_size': [1000],
#              'epochs': [5],
#              'optimizer': ['adam']
#              }

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'recall',
                           cv = 2)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_


# Predicting the Test set results
y_pred_probab = grid_search.predict(X_test)

y_pred = np.zeros(len(y_test))
for i in range(len(y_test)):
    if y_pred_probab[i] > .5:
        y_pred[i] =1
        

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, roc_curve, auc, precision_recall_curve 
from sklearn.metrics import precision_recall_fscore_support
cm = confusion_matrix(y_test, y_pred)

#accuracy -number of instance correctly classified
acsc = accuracy_score(y_test, y_pred) 
df_cm = pd.DataFrame([[cm[1][1], cm[0][0],cm[0][1], cm[1][0]]], 
                        index=[0],
                        columns=['True Positives','True Negatives', 'False Positives', 'False Negatives'])
print(df_cm)
#precision, recall, fscore, support
precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred,average='binary')

#balanced_as = balanced_accuracy_score(y_test, y_pred)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_probab, pos_label=1)
roc_auc = auc(fpr,tpr) # ROC-AUC

#precision recall AUC ->PRC
prc_precision, prc_recall, prc_thresholds = precision_recall_curve(y_test, y_pred_probab)
#prc_auc = auc(prc_precision,prc_recall)
prc_auc = ''
df_metrics = pd.DataFrame([[acsc, precision, recall, fscore,roc_auc]], 
                        index=[0],
                        columns=['accuracy','precision', 'recall', 'fscore', 'ROC-AUC'])

print(df_metrics)


end = time.time()
print(df_metrics.iloc[0][0],',',df_metrics.iloc[0][1],',',df_metrics.iloc[0][2],',',df_metrics.iloc[0][3],',',df_metrics.iloc[0][4],',',df_cm.iloc[0][0],',',df_cm.iloc[0][1],',',df_cm.iloc[0][2],',',df_cm.iloc[0][3],',', end-start)

print("Time taken:", end-start)