# Author: Sheikh Rabiul Islam
# Date: 3/14/2019
# Purpose: preprocess data by running following 4 files
#	data_extract.py -> read the dataset; store it in sqlite database.
#	data_transform.py -> clean data; sample data
#	data_load.py -> move cleaned and sampled data from raw tables to main table. Also write into data/data_preprocessed.csv.
#	data_conversion -> remove unimportant features; labelEncode; onHotEncode;Scale; resample minority class;
#		save the fully processed data as numpy array (binary: data/____.npy)

import time

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
