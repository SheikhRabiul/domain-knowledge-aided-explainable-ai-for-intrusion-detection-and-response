# Author: Sheikh Rabiul Islam
# Date: 7/19/2019
# Purpose: run following algorithms for three different types of configurations (using all features, selected features, and domain features)
import time
s = time.time()

import sys
sys.stdout = open('tmp/exp1.txt', 'w')  #comment this line in case you want to see output on the console.

import pandas as pd
config_file = 'config.txt'

#in the beginning run the classifiers using all features
config = pd.read_csv(config_file,sep=',', index_col =None)
config.iloc[1,1] = 1
config.to_csv(config_file,encoding='utf-8',index=False)
del config
time.sleep(2)


start = time.time()
exec(open("classifier_lr.py").read())
end = time.time()
print("\n\nTime taken by classifier_lr.py:", end-start)

start = time.time()
exec(open("classifier_dt.py").read())
end = time.time()
print("\n\nTime taken by classifier_dt.py:", end-start)

start = time.time()
exec(open("classifier_rf.py").read())
end = time.time()
print("\n\nTime taken by classifier_rf.py:", end-start)

start = time.time()
exec(open("classifier_et.py").read())
end = time.time()
print("\n\nTime taken by classifier_et.py:", end-start)

start = time.time()
exec(open("classifier_gradient_boosting.py").read())
end = time.time()
print("\n\nTime taken by classifier_gradient_boosting.py:", end-start)

start = time.time()
exec(open("classifier_adaboost.py").read())
end = time.time()
print("\n\nTime taken by classifier_adaboost.py:", end-start)

start = time.time()
exec(open("classifier_nb.py").read())
end = time.time()
print("\n\nTime taken by classifier_nb.py:", end-start)

start = time.time()
exec(open("classifier_mda_qda.py").read())
end = time.time()
print("\n\nTime taken by classifier_mda_qda.py:", end-start)

start = time.time()
exec(open("classifier_svm_linear.py").read())
end = time.time()
print("\n\nTime taken by classifier_svm_linear.py:", end-start)

start = time.time()
exec(open("classifier_ann.py").read())
end = time.time()
print("\n\nTime taken by classifier_ann.py:", end-start)



#run the classifiers using selected features
config = pd.read_csv(config_file,sep=',', index_col =None)
config.iloc[1,1] = 2
config.to_csv(config_file,encoding='utf-8',index=False)
del config
time.sleep(2)

start = time.time()
exec(open("classifier_lr.py").read())
end = time.time()
print("\n\nTime taken by classifier_lr.py:", end-start)

start = time.time()
exec(open("classifier_dt.py").read())
end = time.time()
print("\n\nTime taken by classifier_dt.py:", end-start)

start = time.time()
exec(open("classifier_rf.py").read())
end = time.time()
print("\n\nTime taken by classifier_rf.py:", end-start)

start = time.time()
exec(open("classifier_et.py").read())
end = time.time()
print("\n\nTime taken by classifier_et.py:", end-start)

start = time.time()
exec(open("classifier_gradient_boosting.py").read())
end = time.time()
print("\n\nTime taken by classifier_gradient_boosting.py:", end-start)

start = time.time()
exec(open("classifier_adaboost.py").read())
end = time.time()
print("\n\nTime taken by classifier_adaboost.py:", end-start)

start = time.time()
exec(open("classifier_nb.py").read())
end = time.time()
print("\n\nTime taken by classifier_nb.py:", end-start)

start = time.time()
exec(open("classifier_mda_qda.py").read())
end = time.time()
print("\n\nTime taken by classifier_mda_qda.py:", end-start)

start = time.time()
exec(open("classifier_svm_linear.py").read())
end = time.time()
print("\n\nTime taken by classifier_svm_linear.py:", end-start)

start = time.time()
exec(open("classifier_ann.py").read())
end = time.time()
print("\n\nTime taken by classifier_ann.py:", end-start)


#run the classifiers using domain features
config = pd.read_csv(config_file,sep=',', index_col =None)
config.iloc[1,1] = 3
config.to_csv(config_file,encoding='utf-8',index=False)
del config
time.sleep(2)

start = time.time()
exec(open("classifier_lr.py").read())
end = time.time()
print("\n\nTime taken by classifier_lr.py:", end-start)

start = time.time()
exec(open("classifier_dt.py").read())
end = time.time()
print("\n\nTime taken by classifier_dt.py:", end-start)

start = time.time()
exec(open("classifier_rf.py").read())
end = time.time()
print("\n\nTime taken by classifier_rf.py:", end-start)

start = time.time()
exec(open("classifier_et.py").read())
end = time.time()
print("\n\nTime taken by classifier_et.py:", end-start)

start = time.time()
exec(open("classifier_gradient_boosting.py").read())
end = time.time()
print("\n\nTime taken by classifier_gradient_boosting.py:", end-start)

start = time.time()
exec(open("classifier_adaboost.py").read())
end = time.time()
print("\n\nTime taken by classifier_adaboost.py:", end-start)

start = time.time()
exec(open("classifier_nb.py").read())
end = time.time()
print("\n\nTime taken by classifier_nb.py:", end-start)

start = time.time()
exec(open("classifier_mda_qda.py").read())
end = time.time()
print("\n\nTime taken by classifier_mda_qda.py:", end-start)

start = time.time()
exec(open("classifier_svm_linear.py").read())
end = time.time()
print("\n\nTime taken by classifier_svm_linear.py:", end-start)

start = time.time()
exec(open("classifier_ann.py").read())
end = time.time()
print("\n\nTime taken by classifier_ann.py:", end-start)

e = time.time()
print("\n\nTotal Time taken by all classifiers.py:", e-s)



