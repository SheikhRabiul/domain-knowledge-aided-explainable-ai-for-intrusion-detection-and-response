#!/usr/bin/env python
#!/usr/bin/env python 
#Author: Robert martin
#source: https://github.com/rmartinshort/GeneticAlgorithm
#Modified by: Sheikh Rabiul Islam
#Date: 03/19/2018
#A genetic algorithm to aid in feature selection in machine learning problems and then use those selected features in RF to make the prediction. 

from GeneticAlgorithm import GeneticAlgorithm
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import time

start = time.time()

#load traning and test set
file_name_train = os.path.join("data/","data_preprocessed_numerical_train.csv")
file_name_test = os.path.join("data/","data_preprocessed_numerical_test.csv")
df_train = pd.read_csv(file_name_train, sep=',')
df_test = pd.read_csv(file_name_test, sep=',')
training_size = len(df_train)
test_size = len(df_test)

feature_df = pd.concat([df_train,df_test])
feature_df.drop(labels=['Unnamed: 0'], axis=1, inplace= True)
feature_df_bk =feature_df

#save all feature names for future use
file_name_all_features = os.path.join("data/","all_features.csv")
df_all_features = pd.DataFrame(data = feature_df.columns.values, columns = ['feature_name'])
df_all_features.to_csv(file_name_all_features, encoding ='utf-8')


Y = pd.get_dummies(feature_df['defaulted'],drop_first=True)
X = feature_df.drop(['defaulted'],axis=1)


#as it is already shuffled, shufling is off so, last test_size # of whole will be treated as test data.
RF = RandomForestClassifier(n_estimators=1,min_samples_split=2,min_samples_leaf=1)

#call Genetic Algorithm
GA = GeneticAlgorithm(X,Y,RF,test_size)

GA.fit()

best_fitness = GA.best_fitness
print("best_fitness:",best_fitness)

best_individual = GA.best_individual 
#print("best_individual:",best_individual)

best_individual_evolution = GA.best_individual_evolution
#print("best_individual_evolution:",best_individual_evolution)

X_subset = GA.feature_selection
selected_features = list(X_subset.columns.values)
print("Selected Feature names:", selected_features)


file_name_selected_features = os.path.join("data/","selected_features.csv")


df_selected_features = pd.DataFrame(data = selected_features, columns = ['feature_name'])
df_selected_features.to_csv(file_name_selected_features, encoding ='utf-8')


#now apply random forest using only the selected features
#excludes features those were not selected by RF, no need to check last column default
selected_features_list =  df_selected_features['feature_name'].tolist()
all_features_list =  df_all_features['feature_name'].tolist()

deleted_columns = []
for i in range(len(all_features_list)-1):
    if all_features_list[i] not in selected_features_list:
        deleted_columns.append(all_features_list[i])

#print(deleted_columns)

#drop unselected 
feature_df_bk.drop(labels = deleted_columns, inplace=True, axis=1)

X = feature_df_bk.iloc[:,0:-1].values
y = feature_df_bk.iloc[:,-1].values

#Whole list of nominal variables
nominal_l = ['modificationFlag','netSalesProceeds','stepModificationFlag','deferredPaymentModification','firstTimeHomeBuyerFlag','metropolitanDivisionOrMSA', 'occupancyStatus', 'channel','prepaymentPenaltyMortgageFlag','productType','propertyState','propertyType','postalCode','loanPurpose']
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


labelencoder_X = LabelEncoder()

nominal_indexes = []

#get indexes for nominal attributes
nominal_index = []
for i in range(len(selected_features_list)):
    if selected_features_list[i] in nominal_l:
        nominal_index.append(i)
        #encoding response variables
        X[:, i] = labelencoder_X.fit_transform(X[:, i])
# dummy variables
#onehotencoder = OneHotEncoder(categorical_features = nominal_indexes)
#X = onehotencoder.fit_transform(X)

X_df = pd.DataFrame(data=X)
X_df  = pd.get_dummies(X_df, drop_first=True)

X = X_df.values

# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Feature Scaling (scaling all attributes/featues in the same scale)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

#no shufling to treat first portion as training set. It was shuffled before.
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = test_size, shuffle= False, stratify=None)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10,criterion="gini")
# Fitting classifier to the Training set    
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

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

fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(X_test)[:,1], pos_label=1)
roc_auc = auc(fpr,tpr) # ROC-AUC

#precision recall AUC ->PRC
prc_precision, prc_recall, prc_thresholds = precision_recall_curve(y_test, classifier.predict_proba(X_test)[:,1])
#prc_auc = auc(prc_precision,prc_recall)
prc_auc = ''

df_metrics = pd.DataFrame([[acsc, precision, recall, fscore,roc_auc,prc_auc]], 
                        index=[0],
                        columns=['accuracy','precision', 'recall', 'fscore', 'ROC-AUC','PRC-AUC'])

print(df_metrics)


end = time.time()

print(df_metrics.iloc[0][0],',',df_metrics.iloc[0][1],',',df_metrics.iloc[0][2],',',df_metrics.iloc[0][3],',',df_metrics.iloc[0][4],',',df_cm.iloc[0][0],',',df_cm.iloc[0][1],',',df_cm.iloc[0][2],',',df_cm.iloc[0][3],',', end-start)

print("Time taken:", end-start)