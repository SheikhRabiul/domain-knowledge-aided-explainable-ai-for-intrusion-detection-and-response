# Author: Sheikh Rabiul Islam
# Date: 07/19/2019
# Purpose: for each attack, delete all records from training set, run classifier and analyze results for all, selected, and domain features.  


import time
import pandas as pd
s = time.time()

import sys
sys.stdout = open('tmp/exp2.txt', 'w')  #comment this line in case you want to see output on the console.

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
    exec(open("classifier_nb.py").read())
    end = time.time()
    print("\n\nTime taken by  classifier", end-start)
    
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
    exec(open("classifier_nb.py").read())
    end = time.time()
    print("\n\nTime taken by classifier", end-start)
    
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
    exec(open("classifier_nb.py").read())
    end = time.time()
    print("\n\nTime taken by classifier ", end-start)
    
    start = time.time()
    exec(open("analysis_excludeing_1_attack.py").read())
    end = time.time()
    print("\n\nTime taken by [domain] analysis_excludeing_1_attack.py.py:", end-start)

#


