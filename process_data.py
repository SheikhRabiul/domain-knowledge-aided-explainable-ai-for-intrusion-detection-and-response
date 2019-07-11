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
exec(open("data_extract.py").read())
end = time.time()
print("Time taken by data_extract.py:", end-start)

start = time.time()
exec(open("data_transform.py").read())
end = time.time()
print("Time taken by data_transform.py:", end-start)

start = time.time()
exec(open("data_load.py").read())
end = time.time()
print("Time taken by data_load.py:", end-start)

start = time.time()
exec(open("data_conversion.py").read())
end = time.time()
print("Time taken by data_conversion.py:", end-start)

start = time.time()
exec(open("data_conversion_alt.py").read())
end = time.time()
print("Time taken by data_conversion_alt.py:", end-start)