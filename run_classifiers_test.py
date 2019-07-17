# Author: Sheikh Rabiul Islam
# Date: 3/20/2019
# Purpose: run following algorithms; classifiers.ga might not run on windows, it runs on linux machine
#	classifier_lr.py -> Logistic Regression (LR)
#	classifier_dt.py -> Decision Tree (DT)
#	classifier_rf.py -> Random Forest (RF)
#	classifier_et -> Extra Trees (ET)
#	classifier_gradient_boosting.py - > Gradient Boosting (GB)
#	classifier_adaboost.py -> Adaboost
#	classifier_nb.py -> Naive Bayes
#	clasifier_mda.py -> Multiple Discriminant Analysis (MDA)
#	classifier_svm.py -> Support Vector Machine (SVM)
#	classifier_rough_set.py -> Rough Set (RS)
#	classifier_ann.py -> Artificial Neural Network (ANN)
#	classifier_ga.py -> Genetic Algorithm (GA)

import time
import pandas as pd
s = time.time()

import sys
sys.stdout = open('print_exp2_res.txt', 'w')  #comment this line in case you want to see output on the console.

config_file = 'config.txt'

for i in range(1,15):
    start = time.time()
    exec(open("data_preprocess_ex2_all_features.py").read())
    end = time.time()
    print("\n\nTime taken by data_preprocess_ex2_all_features.py:", end-start)
    
    start = time.time()
    exec(open("data_preprocess_ex2_selected_features.py").read())
    end = time.time()
    print("\n\nTime taken by data_preprocess_ex2_selected_features.py:", end-start)
    
    start = time.time()
    exec(open("data_preprocess_ex2_domain_features.py").read())
    end = time.time()
    print("\n\nTime taken by data_preprocess_ex2_domain_features.py:", end-start)
    

    config = pd.read_csv(config_file,sep=',', index_col =None)
    config.iloc[2,1] = i
    config.iloc[1,1] = 1
    config.to_csv(config_file,encoding='utf-8',index=False)
    del config
    time.sleep(2)

    start = time.time()
    exec(open("classifier_et.py").read())
    end = time.time()
    print("\n\nTime taken by classifier_et.py:", end-start)
    
    start = time.time()
    exec(open("analysis_excludeing_1_attack.py").read())
    end = time.time()
    print("\n\nTime taken by [all] analysis_excludeing_1_attack.py.py:", end-start)
    
    
    config = pd.read_csv(config_file,sep=',', index_col =None)
    config.iloc[1,1] = 2
    config.to_csv(config_file,encoding='utf-8',index=False)
    del config
    time.sleep(2)
    
    start = time.time()
    exec(open("classifier_et.py").read())
    end = time.time()
    print("\n\nTime taken by classifier_et.py:", end-start)
    
    start = time.time()
    exec(open("analysis_excludeing_1_attack.py").read())
    end = time.time()
    print("\n\nTime taken by[selected] analysis_excludeing_1_attack.py.py:", end-start)
    
    
    config = pd.read_csv(config_file,sep=',', index_col =None)
    config.iloc[1,1] = 3
    config.to_csv(config_file,encoding='utf-8',index=False)
    del config
    time.sleep(2)
    
    start = time.time()
    exec(open("classifier_et.py").read())
    end = time.time()
    print("\n\nTime taken by classifier_et.py:", end-start)
    
    start = time.time()
    exec(open("analysis_excludeing_1_attack.py").read())
    end = time.time()
    print("\n\nTime taken by [domain] analysis_excludeing_1_attack.py.py:", end-start)

#


