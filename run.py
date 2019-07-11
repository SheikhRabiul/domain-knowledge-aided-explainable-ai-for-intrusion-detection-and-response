# Author: Sheikh Rabiul Islam
# Date: 3/20/2019
import sys
sys.stdout = open('print.txt', 'w')  #comment this line in case you want to see output on the console.

import time
s = time.time()
# one set of run is for original data
# another set of run is using resampled data

start = time.time()
exec(open("process_data.py").read())
end = time.time()
print("\n\nTime taken by process_data.py:", end-start)

start = time.time()
exec(open("run_classifiers.py").read())
end = time.time()
print("Time taken by run_classifiers.py:", end-start)


e = time.time()
print("\n\nTotal Time taken by all classifiers.py:", e-s)


