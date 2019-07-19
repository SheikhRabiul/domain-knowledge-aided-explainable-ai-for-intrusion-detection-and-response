# Author: Sheikh Rabiul Islam
# Date: 07/19/2019
# Purpose: preprocess data by running 3 files for experiment 2; set attack id in config.txt file for the attack need to be deleted from training set.
# data_preprocess_ex2_all_features.py must be run before running other two file as there is some dependency. 
#the preprocessing and 	classifier run  could be done together for all attacks in one place by running run_classifiers_exp2.py 

import time

start = time.time()
exec(open("data_preprocess_ex2_all_features.py.py").read())
end = time.time()
print("Time taken by data_preprocess_ex2_all_features.py:", end-start)

start = time.time()
exec(open("data_preprocess_ex2_selected_features.py").read())
end = time.time()
print("Time taken by data_preprocess_ex2_selected_features.py:", end-start)

start = time.time()
exec(open("data_preprocess_ex2_domain_features.py").read())
end = time.time()
print("Time taken by data_preprocess_ex2_domain_features.py:", end-start)