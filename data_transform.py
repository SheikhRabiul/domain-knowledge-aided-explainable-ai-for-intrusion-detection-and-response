# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 19:39:57 2019

@author: Sheikh Rabiul Islam
Purpose: clean data; sample data (stratified)
"""

#import modules
import pandas as pd   
import numpy as np
import sqlite3
import time

conn = sqlite3.connect("database/credit.sqlite")
curr = conn.cursor()

i=1

#################### labeling data ####################
#label data based on  zeroBalanceCode
#first for all records set default =0
if curr.execute("update performance_raw set defaulted = 0;"):
    print(i, " ok")
else:
    print(i, " failed")
i = i +1
conn.commit()

#then set defaulted =1 for particular rows
if curr.execute("update performance_raw set defaulted = 1 where zeroBalanceCode = 3 or zeroBalanceCode = 6 or zeroBalanceCode = 9;"):
    print(i, " ok")
else:
    print(i, " failed")
i = i +1
conn.commit()

#update this status in origin_raw table too
if curr.execute("update origin_raw set defaulted = 0;"):
    print(i, " ok")
else:
    print(i, " failed")
i = i +1
conn.commit()

#then set defaulted =1 for particular rows
if curr.execute("update origin_raw set defaulted = 1 where loanSequenceNumber in (select loanSequencenumber from performance_raw where defaulted =1);"):
    print(i, " ok")
else:
    print(i, " failed")
i = i +1
conn.commit()


# delete rows with missing values in important attributes
if curr.execute("delete from  origin_raw where  creditScore < 1;"):
    print(i, " ok")
else:
    print(i, " failed")
i = i +1
conn.commit()

if curr.execute("delete from  origin_raw where  originalLoanToValue is Null;"):
    print(i, " ok")
else:
    print(i, " failed")
i = i +1
conn.commit()

if curr.execute("delete from  origin_raw where  originalDebtToIncomeRatio is Null;"):
    print(i, " ok")
else:
    print(i, " failed")
i = i +1
conn.commit()

if curr.execute("delete from  origin_raw where  originalInterestRate is Null;"):
    print(i, " ok")
else:
    print(i, " failed")
i = i +1
conn.commit()

########################## sampling data [stratified] ######################################
#now data is ready for sampling; comment this part if sampling is not needed;
dataset = pd.read_sql_query("select * from origin_raw;", conn)
X = dataset.iloc[:,:].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_remaining, X_sampled, y_remaining, y_sampled = train_test_split( X, y, test_size=2000, random_state=42, stratify=y)

selected_indices = X_sampled[:,19]

dataset_sampled = pd.DataFrame(data=selected_indices, columns = ['loanSequenceNumber'])
dataset_sampled.to_sql("origin_raw_temp", conn, if_exists = "replace", index= True)
conn.commit()

#delete non-sampled accounts from origin_raw table
if curr.execute("delete from origin_raw where loanSequenceNumber not in (select loanSequencenumber from origin_raw_temp);"):
    print(i, " ok")
else:
    print(i, " failed")
i = i +1
conn.commit()

#delete rows for non sampled accounts from performance_raw table
if curr.execute("delete from performance_raw where loanSequenceNumber not in (select loanSequencenumber from origin_raw_temp);"):
    print(i, " ok")
else:
    print(i, " failed")
i = i +1
conn.commit()


####################################### taking care of missing values ################################

#check currentLoanDelinquencyStatus, we are assuming this as numerical, if it has value R then convert that into numeric.
if curr.execute("update  performance_raw set currentLoanDelinquencyStatus = 199 where currentLoanDelinquencyStatus = 'R';"):
    print(i, " ok")
else:
    print(i, " failed")
i = i +1

if curr.execute("update  performance_raw set `zeroBalanceCode`=0 where `zeroBalanceCode` is Null;"):
    print(i, " ok")
else:
    print(i, " failed")

i = i +1

# if NET SALES PROCEEDS = C then it should be replaced by UPB. if it is U , then we cna replace it with null
if curr.execute("update  performance_raw set netSalesProceeds = currentActualUPB where netSalesProceeds = 'C';"):
    print(i, " ok")
else:
    print(i, " failed")
i = i +1

if curr.execute("update  performance_raw set netSalesProceeds = Null where netSalesProceeds = 'U';"):
    print(i, " ok")
else:
    print(i, " failed")
i = i +1
#7   
if curr.execute("update performance_raw set repurchaseFlag = 'NotA' where repurchaseFlag is Null;"):
    print(i, " ok")
else:
    print(i, " failed")
i = i +1
#8
if curr.execute("update performance_raw set modificationFlag = 'NotA' where modificationFlag is Null;"):
    print(i, " ok")
else:
    print(i, " failed")
i = i +1
#9
if curr.execute("update performance_raw set zeroBalanceEffectiveDate = 'NotA' where zeroBalanceEffectiveDate is Null;"):
    print(i, " ok")
else:
    print(i, " failed")
i = i +1
#10
if curr.execute("update performance_raw set dueDateOfLastPaidInstallment = 'NotA' where dueDateOfLastPaidInstallment is Null;"):
    print(i, " ok")
else:
    print(i, " failed")
i = i +1
#11
if curr.execute("update origin_raw set metropolitanDivisionOrMSA = 'NotA' where metropolitanDivisionOrMSA is Null;"):
    print(i, " ok")
else:
    print(i, " failed")
i = i +1
#12
if curr.execute("update origin_raw set prepaymentPenaltyMortgageFlag = 'NotA' where prepaymentPenaltyMortgageFlag is Null;"):
    print(i, " ok")
else:
    print(i, " failed")
i = i +1
#13
if curr.execute("update origin_raw set postalCode = '0' where postalCode is Null;"):
    print(i, " ok")
else:
    print(i, " failed")
i = i +1
#14
if curr.execute("update origin_raw set superConformingFlag = 'NotA' where superConformingFlag is Null;"):
    print(i, " ok")
else:
    print(i, " failed")
i = i +1

#14_1
if curr.execute("update origin_raw set firstTimeHomeBuyerFlag = 'NotA' where firstTimeHomeBuyerFlag = '9';"):
    print(i, " ok")
else:
    print(i, " failed")
i = i +1

if curr.execute("update origin_raw set loanPurpose = 'NotA' where loanPurpose ='9';"):
    print(i, " ok")
else:
    print(i, " failed")
i = i +1


#15
if curr.execute("update origin_raw set preHarpLoanSequenceNumber = 'NotA' where preHarpLoanSequenceNumber is Null;"):
    print(i, " ok")
else:
    print(i, " failed")
i = i +1
#16
if curr.execute("update performance_raw set miRecoveries = 0.00 where miRecoveries is Null;"):
    print(i, " ok")
else:
    print(i, " failed")
i = i +1
#17
if curr.execute("update performance_raw set netSalesProceeds = 'NotA' where netSalesProceeds is Null;"):
    print(i, " ok")
else:
    print(i, " failed")
i = i +1
#18
if curr.execute("update performance_raw set nonMiRecoveries = 0.00 where nonMiRecoveries is Null;"):
    print(i, " ok")
else:
    print(i, " failed")
i = i +1
#19
if curr.execute("update performance_raw set expenses = 0.00 where expenses is Null;"):
    print(i, " ok")
else:
    print(i, " failed")
i = i +1
#20
if curr.execute("update performance_raw set legalCosts = 0.00 where legalCosts is Null;"):
    print(i, " ok")
else:
    print(i, " failed")
i = i +1
#21
if curr.execute("update performance_raw set maintenanceAndPreservationCosts = 0.00 where maintenanceAndPreservationCosts is Null;"):
    print(i, " ok")
else:
    print(i, " failed")
i = i +1
#22
if curr.execute("update performance_raw set taxesAndInsurance = 0.00 where taxesAndInsurance is Null;"):
    print(i, " ok")
else:
    print(i, " failed")
i = i +1
#23
if curr.execute("update performance_raw set miscellaneousExpenses = 0.00 where miscellaneousExpenses is Null;"):
    print(i, " ok")
else:
    print(i, " failed")
i = i +1
#24
if curr.execute("update performance_raw set actualLossCalculation = 0.00 where actualLossCalculation is Null;"):
    print(i, " ok")
else:
    print(i, " failed")
i = i +1
#25
if curr.execute("update performance_raw set modificationCost = 0.00 where modificationCost is Null;"):
    print(i, " ok")
else:
    print(i, " failed")
i = i +1
#26
if curr.execute("update performance_raw set stepModificationFlag = 'NotA' where stepModificationFlag is Null;"):
    print(i, " ok")
else:
    print(i, " failed")
i = i +1
#27
if curr.execute("update performance_raw set deferredPaymentModification = 'NotA' where deferredPaymentModification is Null;"):
    print(i, " ok")
else:
    print(i, " failed")
i = i +1
#28
if curr.execute("update performance_raw set estimatedLoandToValue = 0.0 where estimatedLoandToValue is Null;"):
    print(i, " ok")
else:
    print(i, " failed")

conn.commit()
curr.close()
conn.close()
