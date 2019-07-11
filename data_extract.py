# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 19:39:57 2019

@author: Sheikh Rabiul Islam
Purpose: Data extracttion
    extract data from csv file to raw tables
"""

#import modules
import pandas as pd   
import numpy as np
import sqlite3
import time


file_name1 = 'data/sample_orig_2008.txt'
file_name2 = 'data/sample_svcg_2008.txt'

#year and quarter is unimportant when processing one year at a time. 
year = '2008'
quarter = '1'

conn = sqlite3.connect("database/credit.sqlite")
curr = conn.cursor()

#clear old data from tables
curr.execute("delete from performance_raw;")
conn.commit()
curr.execute("delete from origin_raw;")
conn.commit()

df_origin = pd.read_csv(file_name1,sep="|",header=None, names = ['creditScore','firstPaymentDate','firstTimeHomeBuyerFlag','maturityDate','metropolitanDivisionOrMSA','mortgageInsurancePercentage','numberOfUnits','occupancyStatus','originalCombinedLoanToValue','originalDebtToIncomeRatio','originalUPB','originalLoanToValue','originalInterestRate','channel','prepaymentPenaltyMortgageFlag','productType','propertyState','propertyType','postalCode','loanSequenceNumber','loanPurpose','originalLoanTerm','numberOfBorrowers','sellerName','servicerName','superConformingFlag','preHarpLoanSequenceNumber'])
df_origin['yr'] = year
df_origin['qr'] = quarter
df_origin.to_sql("origin_raw", conn, if_exists = "append", index= False)
conn.commit()

df_performance = pd.read_csv(file_name2,sep="|",header=None, names = ['loanSequenceNumber','monthlyReportingPeriod','currentActualUPB','currentLoanDelinquencyStatus','loanAge','remainingMonthToLegalMaturity','repurchaseFlag','modificationFlag','zeroBalanceCode','zeroBalanceEffectiveDate','currentInterestRate','currentDeferredUPB','dueDateOfLastPaidInstallment','miRecoveries','netSalesProceeds','nonMiRecoveries','expenses','legalCosts','maintenanceAndPreservationCosts','taxesAndInsurance','miscellaneousExpenses','actualLossCalculation','modificationCost','stepModificationFlag','deferredPaymentModification','estimatedLoandToValue'])
df_performance['yr'] = year
df_performance['qr'] = quarter
df_performance.to_sql("performance_raw", conn, if_exists = "append", index= False)
conn.commit()

curr.close()
conn.close()
