# Author: Sheikh Rabiul Islam
# Date: 07/19/2019
# Purpose: preprocess data by running following 5 files for experiment 1
#    data_merger.py -> merge individual csv files in to a single file following same format (same column structure).
#    data_sampler.py -> sample specified number of records using stratified sampling technique. 
#    data_preprocess_all_features.py -> preprocess data using all features, saves fully preprocessed data in binary format (as numpy array with format .npy)
#    data_preprocess_selected_features.py -> preprocess data using selected features, saves fully preprocessed data in binary format (as numpy array with format .npy)
#    data_preprocess_domain_features.py -> preprocess data using domain features, saves fully preprocessed data in binary format (as numpy array with format .npy)
	
import time

start = time.time()
exec(open("data_merger.py").read())
end = time.time()
print("Time taken by data_merger.py:", end-start)

start = time.time()
exec(open("data_sampler.py").read())
end = time.time()
print("Time taken by data_sampler.py:", end-start)

start = time.time()
exec(open("data_preprocess_all_features.py").read())
end = time.time()
print("Time taken by data_preprocess_all_features.py:", end-start)

start = time.time()
exec(open("data_preprocess_selected_features.py").read())
end = time.time()
print("Time taken by data_preprocess_selected_features.py:", end-start)

start = time.time()
exec(open("data_preprocess_domain_features.py").read())
end = time.time()
print("Time taken by data_preprocess_domain_features.py:", end-start)
