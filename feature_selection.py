# Author: Sheikh Rabiul Islam
# Date: 02/10/2019
# Purpose: 

#import modules
import pandas as pd   
import numpy as np
import time
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score, ShuffleSplit
from collections import defaultdict
from sklearn.metrics import r2_score

output_folder = "output/"

# import data
dataset = pd.read_csv('data/result.csv', sep=',')


#move default column to the end and delete unnecessary columns.
defaulted = dataset['defaulted']
dataset.drop(labels=['defaulted','currentLoanDelinquencyStatus','Unnamed: 0','loanSequenceNumber','loanSequenceNumber.1','sellerName','servicerName','yr.1','qr.1','preHarpLoanSequenceNumber'], axis=1, inplace = True)
dataset.insert(len(dataset.columns),'defaulted', defaulted)


# seperate the dependent (target) variaable
X = dataset.iloc[:,0:-1].values
X_y =dataset.iloc[:,0:-1]
X_columns = dataset.iloc[:,0:-1].columns.values
y = dataset.iloc[:,-1].values

X_cloumns_d = { X_columns[i]:i for i in range(0, len(X_columns)) }

for i in range(0,len(X_columns)):
    print(i,',',X_columns[i])
    
"""
0 , monthlyReportingPeriod
1 , currentActualUPB
2 , loanAge
3 , remainingMonthToLegalMaturity
4 , repurchaseFlag
5 , modificationFlag
6 , zeroBalanceCode
7 , zeroBalanceEffectiveDate
8 , currentInterestRate
9 , currentDeferredUPB
10 , dueDateOfLastPaidInstallment
11 , miRecoveries
12 , netSalesProceeds
13 , nonMiRecoveries
14 , expenses
15 , legalCosts
16 , maintenanceAndPreservationCosts
17 , taxesAndInsurance
18 , miscellaneousExpenses
19 , actualLossCalculation
20 , modificationCost
21 , stepModificationFlag
22 , deferredPaymentModification
23 , estimatedLoandToValue
24 , yr
25 , qr
26 , creditScore
27 , firstPaymentDate
28 , firstTimeHomeBuyerFlag
29 , maturityDate
30 , metropolitanDivisionOrMSA
31 , mortgageInsurancePercentage
32 , numberOfUnits
33 , occupancyStatus
34 , originalCombinedLoanToValue
35 , originalDebtToIncomeRatio
36 , originalUPB
37 , originalLoanToValue
38 , originalInterestRate
39 , channel
40 , prepaymentPenaltyMortgageFlag
41 , productType
42 , propertyState
43 , propertyType
44 , postalCode
45 , loanPurpose
46 , originalLoanTerm
47 , numberOfBorrowers
48 , superConformingFlag
"""

# Encoding categorical data

nominal_l = ['monthlyReportingPeriod','repurchaseFlag', 'modificationFlag', 'zeroBalanceEffectiveDate', 'dueDateOfLastPaidInstallment','netSalesProceeds','stepModificationFlag','deferredPaymentModification', 'yr', 'qr','firstPaymentDate','firstTimeHomeBuyerFlag','maturityDate','metropolitanDivisionOrMSA', 'occupancyStatus', 'channel','prepaymentPenaltyMortgageFlag','productType','propertyState','propertyType','postalCode','loanPurpose','superConformingFlag']
ordinal_l = [ ] # we don't have any feature that requires to preserve order after encoding.

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()

#print(X[:, 5])

nominal_indexes = []
for j in range(0,len(nominal_l)):
    i = X_cloumns_d[nominal_l[j]]
    nominal_indexes.append(i)
    print("executing ",nominal_l[j], " i:",i)
    X[:, i] = labelencoder_X.fit_transform(X[:, i])

df_dump_part1 = pd.DataFrame(X, columns=X_columns)
df_dump_part2 = pd.DataFrame(y, columns=['defaulted'])   
df_dump = pd.concat([df_dump_part1,df_dump_part2], axis = 1)     
df_dump.to_csv("data/result_numeric.csv",encoding='utf-8')
#print(X)
onehotencoder = OneHotEncoder(categorical_features = nominal_indexes)
X = onehotencoder.fit_transform(X).toarray()

#print(X)


# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Feature Scaling (scaling all attributes/featues in the same scale)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


#configure here
selected_algorithm =1 # 1= extra trees, 2 = random Forest
biased = True    # make it false if you want it balanced
threshold_feature_score = 0


def scale_a_number(inpt, to_min, to_max, from_min, from_max):
    return (to_max-to_min)*(inpt-from_min)/(from_max-from_min)+to_min

def scale_a_list(l, to_min, to_max):
    return [scale_a_number(i, to_min, to_max, min(l), max(l)) for i in l]

if selected_algorithm == 1:
    ############ Extra Trees Classifiers
    #balanced
    forest = ExtraTreesClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=False, oob_score=False, n_jobs=10, random_state=None, verbose=0, warm_start=False, class_weight='balanced')
    #biased to malicious class
    if biased:
        forest = ExtraTreesClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=False, oob_score=False, n_jobs=10, random_state=None, verbose=0, warm_start=False, class_weight={0:.9999,1:.0001})
    
    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
	

	
    data= (sorted(zip(map(lambda x: round(x, 4), forest.feature_importances_), X_columns),reverse=True))
    #converting the list to a datafreame
    df_result_et = pd.DataFrame(data,columns=['score','feature'])
    df_result_et.insert(1,"scaled_score", scale_a_list(df_result_et['score'],0,1))
    print("top 5 features using Extra Trees:")
    print(df_result_et.head())
    f_name = 'ranked_features_et.csv'
    f_path = os.path.join(output_folder, f_name)
    print("saving  ranked features in ")
    print(f_path)
    df_result_et.to_csv(f_path, sep=',')

############ Random Forest Classifier)
if selected_algorithm == 2:
    #balanced
    forest = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=4, random_state=None, verbose=0, warm_start=False, class_weight='balanced')
    #biased
    if biased:
        forest = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=4, random_state=None, verbose=0, warm_start=False, class_weight={0:.9999,1:.0001 })
    
    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
    data= (sorted(zip(map(lambda x: round(x, 4), forest.feature_importances_), X_columns),reverse=True))
    #converting the list to a datafreame
    df_result_rfc = pd.DataFrame(data,columns=['score','feature'])
    df_result_rfc.insert(1,"scaled_score", scale_a_list(df_result_rfc['score'],0,1))
    print("top 5 features using Random forest classifier:")
    print(df_result_rfc.head())
    f_name = 'ranked_features_rf.csv'
    f_path = os.path.join(output_folder, f_name)
    print("saving  ranked features in ")
    print(f_path)
    df_result_rfc.to_csv(f_path, sep=',')
    
print("******************* ending feature selection ********************")


print("\n ******************* discarding  insignificant features (score 0) ********************")

if selected_algorithm == 1:    
    df_result_et_new = df_result_et[df_result_et['scaled_score']>0]
    f_name = 'selected_features_et.csv'
    f_path = os.path.join(output_folder, f_name)
    print("saving  filtered features features in ")
    print(f_path)
    df_result_et_new.to_csv(f_path, sep=',')
    
if selected_algorithm == 2:
    df_result_rfc_new = df_result_rfc[df_result_rfc['scaled_score']>0]
    f_name = 'selected_features_rf.csv'
    f_path = os.path.join(output_folder, f_name)
    print("saving  filtered features features in ")
    print(f_path)
    df_result_rfc_new.to_csv(f_path, sep=',')


"""

## apply different classifer from below, uncomment the one you like to be in action
from sklearn.model_selection import KFold, cross_val_score
classifier=''
y_pred_all=np.empty(shape=(0)) #empty 1d numpy array
proba_all=np.empty(shape=(0,2)) # empty 2d numpy array-> o rows 2 column

## SVM
#from sklearn.svm import SVC
#classifier = SVC(kernel = 'linear', random_state = 0, probability=True)

## random forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10,criterion="gini")

##    Naive Bayes
#from sklearn.naive_bayes import BernoulliNB
#classifier = BernoulliNB() 

# gradient Boosting
#from sklearn.ensemble import GradientBoostingClassifier
#classifier = GradientBoostingClassifier()

# knn
#from sklearn.neighbors import KNeighborsClassifier
#classifier = KNeighborsClassifier()

# Extra Trees 
#from sklearn.ensemble import ExtraTreesClassifier
#classifier = ExtraTreesClassifier(criterion="entropy")

##we found extra Trees classifier was the best classifier with highest accuracy.To see other classifiers result 
#keep that two line of code uncommented 


# k fold cross validation

k_fold = KFold(n_splits=10)
start = time.time()
for train_indices, test_indices in k_fold.split(X):
    #print('Train: %s | test: %s' % (train_indices, test_indices))      
    X_train = X[train_indices[0]:train_indices[-1]+1]
    y_train = y[train_indices[0]:train_indices[-1]+1]

    X_test = X[test_indices[0]:test_indices[-1]+1]
    y_test = y[test_indices[0]:test_indices[-1]+1]

    # Fitting SVM to the Training set    
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    y_pred_all=np.concatenate((y_pred_all,y_pred),axis=0)

    proba = classifier.predict_proba(X_test)
    proba_all=np.concatenate((proba_all,proba))
    
end = time.time()
diff = end - start
print("classification time:")
print(diff)
# this gives us how strong is the ye/no decision with a probability value (continuous value 
# rather than just the discrete binary decision)
df_result = pd.DataFrame(proba_all,columns=['probability_no','probability_yes'])


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score
from sklearn.metrics import precision_recall_fscore_support
cm = confusion_matrix(y, y_pred_all)

#accuracy -number of instance correctly classified
acsc = accuracy_score(y, y_pred_all) 
df_cm = pd.DataFrame([[cm[1][1], cm[0][0],cm[0][1], cm[1][0]]], 
                        index=[0],
                        columns=['True Positives','True Negatives', 'False Positives', 'False Negatives'])
print(df_cm)
#precision, recall, fscore, support
precision, recall, fscore, support = precision_recall_fscore_support(y, y_pred_all,average='binary')

df_metrics = pd.DataFrame([[acsc, precision, recall, fscore]], 
                        index=[0],
                        columns=['accuracy','precision', 'recall', 'fscore'])

print(df_metrics)
"""