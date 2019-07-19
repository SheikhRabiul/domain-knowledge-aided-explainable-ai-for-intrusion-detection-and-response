# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 19:39:57 2019

@author: Sheikh Rabiul Islam
Purpose: Create database
    Create necessary tables
"""

#import modules
import pandas as pd   
import numpy as np
import sqlite3
import time

conn = sqlite3.connect("database/credit.sqlite")
curr = conn.cursor()

#create raw tables

curr.execute("CREATE TABLE `origin_raw` (\
	`creditScore`	integer,\
	`firstPaymentDate`	text,\
	`firstTimeHomeBuyerFlag`	text,\
	`maturityDate`	text,\
	`metropolitanDivisionOrMSA`	text,\
	`mortgageInsurancePercentage`	integer,\
	`numberOfUnits`	integer,\
	`occupancyStatus`	text,\
	`originalCombinedLoanToValue`	REAL,\
	`originalDebtToIncomeRatio`	integer,\
	`originalUPB`	integer,\
	`originalLoanToValue`	integer,\
	`originalInterestRate`	REAL,\
	`channel`	text,\
	`prepaymentPenaltyMortgageFlag`	text,\
	`productType`	text,\
	`propertyState`	text,\
	`propertyType`	text,\
	`postalCode`	text,\
	`loanSequenceNumber`	text,\
	`loanPurpose`	text,\
	`originalLoanTerm`	integer,\
	`numberOfBorrowers`	integer,\
	`sellerName`	text,\
	`servicerName`	text,\
	`superConformingFlag`	text,\
	`preHarpLoanSequenceNumber`	text,\
	`yr`	text,\
	`qr`	text,\
	`defaulted`	integer,\
	PRIMARY KEY(`loanSequenceNumber`));")

conn.commit()

curr.execute("CREATE TABLE `performance_raw` (\
	`loanSequenceNumber`	text,\
	`monthlyReportingPeriod`	text,\
	`currentActualUPB`	real,\
	`currentLoanDelinquencyStatus`	integer,\
	`loanAge`	integer,\
	`remainingMonthToLegalMaturity`	integer,\
	`repurchaseFlag`	text,\
	`modificationFlag`	text,\
	`zeroBalanceCode`	integer,\
	`zeroBalanceEffectiveDate`	text,\
	`currentInterestRate`	real,\
	`currentDeferredUPB`	integer,\
	`dueDateOfLastPaidInstallment`	text,\
	`miRecoveries`	real,\
	`netSalesProceeds`	text,\
	`nonMiRecoveries`	real,\
	`expenses`	real,\
	`legalCosts`	real,\
	`maintenanceAndPreservationCosts`	real,\
	`taxesAndInsurance`	real,\
	`miscellaneousExpenses`	real,\
	`actualLossCalculation`	real,\
	`modificationCost`	real,\
	`stepModificationFlag`	text,\
	`deferredPaymentModification`	text,\
	`estimatedLoandToValue`	real,\
	`yr`	text,\
	`qr`	text,\
	`defaulted`	integer,\
	FOREIGN KEY(`loanSequenceNumber`) REFERENCES `origin_raw`(`loanSequenceNumber`));")
conn.commit()

#create orignial tables
curr.execute("CREATE TABLE `origin` (\
	`creditScore`	integer,\
	`firstPaymentDate`	text,\
	`firstTimeHomeBuyerFlag`	text,\
	`maturityDate`	text,\
	`metropolitanDivisionOrMSA`	text,\
	`mortgageInsurancePercentage`	integer,\
	`numberOfUnits`	integer,\
	`occupancyStatus`	text,\
	`originalCombinedLoanToValue`	REAL,\
	`originalDebtToIncomeRatio`	integer,\
	`originalUPB`	integer,\
	`originalLoanToValue`	integer,\
	`originalInterestRate`	REAL,\
	`channel`	text,\
	`prepaymentPenaltyMortgageFlag`	text,\
	`productType`	text,\
	`propertyState`	text,\
	`propertyType`	text,\
	`postalCode`	text,\
	`loanSequenceNumber`	text,\
	`loanPurpose`	text,\
	`originalLoanTerm`	integer,\
	`numberOfBorrowers`	integer,\
	`sellerName`	text,\
	`servicerName`	text,\
	`superConformingFlag`	text,\
	`preHarpLoanSequenceNumber`	text,\
	`yr`	text,\
	`qr`	text,\
	PRIMARY KEY(`loanSequenceNumber`));")

conn.commit()

curr.execute("CREATE TABLE `performance` (\
	`loanSequenceNumber`	text,\
	`monthlyReportingPeriod`	text,\
	`currentActualUPB`	real,\
	`currentLoanDelinquencyStatus`	integer,\
	`loanAge`	integer,\
	`remainingMonthToLegalMaturity`	integer,\
	`repurchaseFlag`	text,\
	`modificationFlag`	text,\
	`zeroBalanceCode`	integer,\
	`zeroBalanceEffectiveDate`	text,\
	`currentInterestRate`	real,\
	`currentDeferredUPB`	integer,\
	`dueDateOfLastPaidInstallment`	text,\
	`miRecoveries`	real,\
	`netSalesProceeds`	text,\
	`nonMiRecoveries`	real,\
	`expenses`	real,\
	`legalCosts`	real,\
	`maintenanceAndPreservationCosts`	real,\
	`taxesAndInsurance`	real,\
	`miscellaneousExpenses`	real,\
	`actualLossCalculation`	real,\
	`modificationCost`	real,\
	`stepModificationFlag`	text,\
	`deferredPaymentModification`	text,\
	`estimatedLoandToValue`	real,\
	`yr`	text,\
	`qr`	text,\
	`defaulted`	integer,\
	FOREIGN KEY(`loanSequenceNumber`) REFERENCES `origin`(`loanSequenceNumber`));")
conn.commit()

curr.close()
conn.close()
