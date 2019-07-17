# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 16:57:36 2019

@author: user1
"""
import pandas as pd
import time
config_file = 'config.txt'
for i in range(5,8):
    attack_id = i 
    config = pd.read_csv(config_file,sep=',', index_col =None)
    config.iloc[2,1] = attack_id
    config.iloc[1,1] = 1
    print("attack id: ", config.iloc[2,1])
    config.to_csv(config_file,encoding='utf-8',index=False)
    del config
    time.sleep(2)
    exec(open("classifier_et.py").read())