# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 19:39:57 2019

@author: Sheikh Rabiul Islam
Purpose: Data load
    load cleaned and extracted data from raw table to original table
"""

#import modules
import pandas as pd   
import numpy as np
import sqlite3
import time

conn = sqlite3.connect("database/credit.sqlite")
curr = conn.cursor()

# copy data
curr.execute("delete from origin;")
curr.execute("delete from performance;")
conn.commit()
curr.execute("insert into origin(creditScore,firstPaymentDate,firstTimeHomeBuyerFlag,maturityDate,metropolitanDivisionOrMSA,mortgageInsurancePercentage,numberOfUnits,occupancyStatus,originalCombinedLoanToValue,originalDebtToIncomeRatio,originalUPB,originalLoanToValue,originalInterestRate,channel,prepaymentPenaltyMortgageFlag,productType,propertyState,propertyType,postalCode,loanSequenceNumber,loanPurpose,originalLoanTerm,numberOfBorrowers,sellerName,servicerName,superConformingFlag,preHarpLoanSequenceNumber,yr,qr) select creditScore,firstPaymentDate,firstTimeHomeBuyerFlag,maturityDate,metropolitanDivisionOrMSA,mortgageInsurancePercentage,numberOfUnits,occupancyStatus,originalCombinedLoanToValue,originalDebtToIncomeRatio,originalUPB,originalLoanToValue,originalInterestRate,channel,prepaymentPenaltyMortgageFlag,productType,propertyState,propertyType,postalCode,loanSequenceNumber,loanPurpose,originalLoanTerm,numberOfBorrowers,sellerName,servicerName,superConformingFlag,preHarpLoanSequenceNumber,yr,qr from origin_raw;")
conn.commit()
curr.execute("insert into performance(loanSequenceNumber,monthlyReportingPeriod,currentActualUPB,currentLoanDelinquencyStatus,loanAge,remainingMonthToLegalMaturity,repurchaseFlag,modificationFlag,zeroBalanceCode,zeroBalanceEffectiveDate,currentInterestRate,currentDeferredUPB,dueDateOfLastPaidInstallment,miRecoveries,netSalesProceeds,nonMiRecoveries,expenses,legalCosts,maintenanceAndPreservationCosts,taxesAndInsurance,miscellaneousExpenses,actualLossCalculation,modificationCost,stepModificationFlag,deferredPaymentModification,estimatedLoandToValue,yr,qr,defaulted) select loanSequenceNumber,monthlyReportingPeriod,currentActualUPB,currentLoanDelinquencyStatus,loanAge,remainingMonthToLegalMaturity,repurchaseFlag,modificationFlag,zeroBalanceCode,zeroBalanceEffectiveDate,currentInterestRate,currentDeferredUPB,dueDateOfLastPaidInstallment,miRecoveries,netSalesProceeds,nonMiRecoveries,expenses,legalCosts,maintenanceAndPreservationCosts,taxesAndInsurance,miscellaneousExpenses,actualLossCalculation,modificationCost,stepModificationFlag,deferredPaymentModification,estimatedLoandToValue,yr,qr,defaulted from performance_raw;")
conn.commit()

df_result = pd.read_sql_query("select a.*,b.* from performance a, origin b  where a.loanSequenceNumber=b.loanSequenceNumber;", conn)
df_result.to_csv("data/data_preprocessed.csv",encoding='utf-8')

curr.close()
conn.close()

