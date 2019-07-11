# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 10:33:25 2019

@author: Sheikh Rabiul Islam
"""
import os
import sys
from shutil import copyfile
from sys import exit

folder_source = "data/2002/"
folder_target = "data/"

file_name = "data_fully_processed_X_train.npy"
source = os.path.join(folder_source, file_name) 
target = os.path.join(folder_target, file_name) 
try:
    copyfile(source, target)
except IOError as e:
    print("Unable to copy file. %s" % e)
    exit(1)
except:
    print("Unexpected error:", sys.exc_info())
    exit(1)
print("\nFile copy done for:",file_name)


file_name = "data_fully_processed_y_train.npy"
source = os.path.join(folder_source, file_name) 
target = os.path.join(folder_target, file_name) 
try:
    copyfile(source, target)
except IOError as e:
    print("Unable to copy file. %s" % e)
    exit(1)
except:
    print("Unexpected error:", sys.exc_info())
    exit(1)
print("\nFile copy done for:",file_name)


file_name = "data_fully_processed_X_test.npy"
source = os.path.join(folder_source, file_name) 
target = os.path.join(folder_target, file_name) 
try:
    copyfile(source, target)
except IOError as e:
    print("Unable to copy file. %s" % e)
    exit(1)
except:
    print("Unexpected error:", sys.exc_info())
    exit(1)
print("\nFile copy done for:",file_name)


file_name = "data_fully_processed_y_test.npy"
source = os.path.join(folder_source, file_name) 
target = os.path.join(folder_target, file_name) 
try:
    copyfile(source, target)
except IOError as e:
    print("Unable to copy file. %s" % e)
    exit(1)
except:
    print("Unexpected error:", sys.exc_info())
    exit(1)
print("\nFile copy done for:",file_name)

file_name = "data_fully_processed_X_train_resampled.npy"
source = os.path.join(folder_source, file_name) 
target = os.path.join(folder_target, file_name) 
try:
    copyfile(source, target)
except IOError as e:
    print("Unable to copy file. %s" % e)
    exit(1)
except:
    print("Unexpected error:", sys.exc_info())
    exit(1)
print("\nFile copy done for:",file_name)

file_name = "data_fully_processed_y_train_resampled.npy"
source = os.path.join(folder_source, file_name) 
target = os.path.join(folder_target, file_name) 
try:
    copyfile(source, target)
except IOError as e:
    print("Unable to copy file. %s" % e)
    exit(1)
except:
    print("Unexpected error:", sys.exc_info())
    exit(1)
print("\nFile copy done for:",file_name)