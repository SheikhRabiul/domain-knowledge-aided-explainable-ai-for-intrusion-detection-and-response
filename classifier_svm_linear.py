# Author: Sheikh Rabiul Islam
# Date: 07/18/2019
# Purpose: Random Forest on fully processed data

#import modules
import pandas as pd   
import numpy as np
import time

#configurations
config_file = 'config.txt'
config = pd.read_csv(config_file,sep=',', index_col =None)
resample_data = config.iloc[0,1] #0 or 1
feature_set = config.iloc[1,1] # 1 = full features, 2 = selected, 3 = domain
attack_id = config.iloc[2,1]
del config

print("SVM:",resample_data)
start = time.time()

from sklearn.model_selection import KFold, cross_val_score

from sklearn.svm import SVC,NuSVC,LinearSVC
#classifier = SVC(kernel = 'linear', random_state = 0, probability=True)
classifier = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)
#from sklearn import svm
#classifier = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto_deprecated', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)
# import processed data
f_X_train = 'data/data_fully_processed_X_train'
f_y_train = 'data/data_fully_processed_y_train'
f_X_test = 'data/data_fully_processed_X_test'
f_y_test = 'data/data_fully_processed_y_test'

if resample_data == 1:
    f_X_train = f_X_train + "_resampled"
    f_y_train = f_y_train + "_resampled"

if feature_set == 1:
    f_X_train = f_X_train + "_all_features"
    f_y_train = f_y_train + "_all_features"
    f_X_test = f_X_test + "_all_features"
    f_y_test = f_y_test + "_all_features"
elif feature_set == 2:
    f_X_train = f_X_train + "_selected_features"
    f_y_train = f_y_train + "_selected_features"
    f_X_test = f_X_test + "_selected_features"
    f_y_test = f_y_test + "_selected_features"
else:
    f_X_train = f_X_train + "_domain_features"
    f_y_train = f_y_train + "_domain_features"
    f_X_test = f_X_test + "_domain_features"
    f_y_test = f_y_test + "_domain_features"  

    

f_X_train = f_X_train + ".npy"
f_y_train = f_y_train + ".npy"
f_X_test = f_X_test + ".npy"
f_y_test = f_y_test + ".npy"


print(f_X_train)
print(f_y_train)
print(f_X_test)
print(f_y_test)

X_train = np.load(f_X_train)
y_train = np.load(f_y_train)
X_test = np.load(f_X_test)
y_test = np.load(f_y_test)

# Fitting classifier to the Training set    
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
#dump y_pred for future use ( to calculate percent of attack detected in case of experiment 2 where we exclude one attack from training set)
df_y_pred = pd.DataFrame(y_pred, columns=['y_pred'])
df_y_pred.to_csv("data/y_pred.csv",encoding='utf-8')   

# Performance metrics
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
#sample code for feature scalling: feature ranking result is scalled in the range of  0 to 1
def scale_a_number(inpt, to_min, to_max, from_min, from_max):
    return (to_max-to_min)*(inpt-from_min)/(from_max-from_min)+to_min
def scale_a_list(l, to_min, to_max):
    return [scale_a_number(i, to_min, to_max, min(l), max(l)) for i in l]

predict_probab = scale_a_list(classifier.decision_function(X_test),0,1)


fpr, tpr, thresholds = roc_curve(y_test, predict_probab)
roc_auc = auc(fpr,tpr) # ROC-AUC

#precision recall AUC ->PRC
prc_precision, prc_recall, prc_thresholds = precision_recall_curve(y_test, predict_probab)
#prc_auc = auc(prc_precision,prc_recall)
prc_auc = ''
df_metrics = pd.DataFrame([[acsc, precision, recall, fscore,roc_auc]], 
                        index=[0],
                        columns=['accuracy','precision', 'recall', 'fscore', 'ROC-AUC'])

print(df_metrics)
end = time.time()

print(df_metrics.iloc[0][0],',',df_metrics.iloc[0][1],',',df_metrics.iloc[0][2],',',df_metrics.iloc[0][3],',',df_metrics.iloc[0][4],',',df_cm.iloc[0][0],',',df_cm.iloc[0][1],',',df_cm.iloc[0][2],',',df_cm.iloc[0][3],',', end-start)

print("Time taken:", end-start)