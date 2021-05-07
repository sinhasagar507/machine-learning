"""
@author: Sagar Sinha

"""


#from google.colab import drive
#drive.mount('/content/drive')

#Import libraries and fixing the random state to 42 for entire notebook

#Set the seed value
seed_value=15042

#Set the seed as a local environment variable
import os
os.environ['PYTHONHASHSEED'] = str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

import warnings
warnings.filterwarnings('ignore')
import joblib, jsonpickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.feature_selection import VarianceThreshold

sns.set_context("paper", font_scale = 1, rc={"grid.linewidth": 3})
pd.set_option('display.max_rows', 100, 'display.max_columns', 400)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRFClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier

df_train = pd.read_csv("B:/ML-NLP/Projects/lTFsdeploy/lTFs-finance/datasets/train_Data.csv", parse_dates=['DisbursalDate','MaturityDAte','AuthDate'])

df_test = pd.read_csv("B:/ML-NLP/Projects/lTFsdeploy/lTFs-finance/datasets/test_Data.csv", parse_dates=['DisbursalDate','MaturityDAte','AuthDate'])


types = {"ID":"int32","WRITE-OFF-AMT":"float32","TENURE":"float16"}
df_train_bureau = pd.read_csv("B:/ML-NLP/Projects/lTFsdeploy/lTFs-finance/datasets/train_bureau.csv",dtype=types,parse_dates=['DATE-REPORTED','DISBURSED-DT','CLOSE-DT','LAST-PAYMENT-DATE'])

types = {"ID":"int32","WRITE-OFF-AMT":"float32","TENURE":"float16"}
df_test_bureau = pd.read_csv("B:/ML-NLP/Projects/lTFsdeploy/lTFs-finance/datasets/test_bureau.csv",dtype=types,parse_dates=['DATE-REPORTED','DISBURSED-DT','CLOSE-DT','LAST-PAYMENT-DATE'])


Submission = df_test[['ID']]

#Combining the train and test datasets for better EDA
Y = df_train['Top-up Month']
df_train.drop(['Top-up Month'], axis=1, inplace=True)
df = pd.concat([df_train, df_test],axis=0)
df.shape
df = df.drop_duplicates()

df.head()

"""**EDA - Bureau Data**

1. Data-Cleaning
2. Feature Engineering and Feature Selection
(Starting with null value imputation)
3. Encoding and further

"""

#Dropping duplicates, if any
df_train_bureau = df_train_bureau.drop_duplicates()
df_test_bureau = df_test_bureau.drop_duplicates()
df_train_bureau['Source'] = "Train"
df_test_bureau['Source'] = "Test"
df_bureau_data = pd.concat([df_train_bureau, df_test_bureau],axis=0)

df_bureau_data.info(memory_usage='deep')

"""  
  1. We can see there are 18 variables that are of object dtype and few are of datetime dtype followed by 1 each in float and int category. 
  2. We can also see that there are many variables with missing values. If the percentage of missing values is more than 50%, we will go ahead and delete the variable. 
  3. We also have to check the object variables, about the cardinality. If there are many values, we might have to combine a few of them.
  
"""

#Displaying initial few rows of data
df_bureau_data.head()

#Displaying the number of null values
(df_bureau_data.isna().sum()/len(df_bureau_data))*100

#Printing the no. of unique columns in each column
for col in df_bureau_data.columns:
    if df_bureau_data[col].dtype == "object":
        print ("Value Count of {} column are:\n{}".format(col, df_bureau_data[col].value_counts(dropna=False)))

#We will drop the MATCH-TYPE column as it has one value which is dominating
df_bureau_data.drop("MATCH-TYPE",axis=1,inplace=True)

#We will combine the values of certain columns which are few in number
df_bureau_data['ACCT-TYPE'].replace({"Secured Credit Card":"Others",
                                  "Business Non-Funded Credit Facility General":"Others",
                                  "Prime Minister Jaan Dhan Yojana - Overdraft":"Others",
                                  "Pradhan Mantri Awas Yojana - CLSS":"Others",
                                  "SHG Individual":"Others",
                                  "JLG Group":"Others",
                                  "Microfinance Personal Loan":"Others",
                                  "Fleet Card":"Others",
                                  "Microfinance Housing Loan":"Others",
                                  "Commercial Equipment Loan":"Others",
                                  "Corporate Credit Card":"Others",
                                  "Loan on Credit Card":"Others",
                                  "Business Non-Funded Credit Facility-Priority Sector-Others":"Others",
                                  "Leasing":"Others",
                                  "Telco Landline":"Others",
                                  "SHG Group":"Others",
                                  "Staff Loan":"Others"},inplace=True)
df_bureau_data['CONTRIBUTOR-TYPE'].replace({"SFB":"Others",
                                         "ARC":"Others",
                                         "OFI":"Others"},inplace=True)
df_bureau_data['ACCOUNT-STATUS'].replace({"SUIT FILED (WILFUL DEFAULT)":"Others",
                                       "WILFUL DEFAULT":"Others",
                                       "Sold/Purchased":"Others",
                                       "Cancelled":"Others"},inplace=True)

# We will drop the ASSET_CLASS variable as it has majority of missing values.
df_bureau_data.drop("ASSET_CLASS", axis=1, inplace=True)

df_bureau_data.isna().sum()/len(df_bureau_data)

df_bureau_data.drop(['TENURE','OVERDUE-AMT','INSTALLMENT-FREQUENCY','CREDIT-LIMIT/SANC AMT','LAST-PAYMENT-DATE','CLOSE-DT'],axis=1,inplace=True)

#We will drop the Date Variables as of now
df_bureau_data.drop(['DATE-REPORTED','DISBURSED-DT','REPORTED DATE - HIST'],axis=1,inplace=True)

#Also drop the Write OFF Amt column
df_bureau_data.drop(df_bureau_data[df_bureau_data['WRITE-OFF-AMT']<0].index,axis=0,inplace=True)

#Its output should be an empty dataframe
df_bureau_data[df_bureau_data['WRITE-OFF-AMT']<0]

#Some of the numerical variables are coded as of object type because of the comma in the numbers, we convert them to the right data type.
df_bureau_data['DISBURSED-AMT/HIGH CREDIT'] = df_bureau_data['DISBURSED-AMT/HIGH CREDIT'].str.replace(",","")
df_bureau_data['DISBURSED-AMT/HIGH CREDIT'] = pd.to_numeric(df_bureau_data['DISBURSED-AMT/HIGH CREDIT'])

df_bureau_data['CURRENT-BAL'] = df_bureau_data['CURRENT-BAL'].str.replace(",","")
df_bureau_data['CURRENT-BAL'] = pd.to_numeric(df_bureau_data['CURRENT-BAL'])

df_bureau_data['DISBURSED-AMT/HIGH CREDIT'].fillna(df_bureau_data['DISBURSED-AMT/HIGH CREDIT'].mean(),inplace=True)
df_bureau_data['CURRENT-BAL'].fillna(df_bureau_data['CURRENT-BAL'].mean(),inplace=True)
df_bureau_data['WRITE-OFF-AMT'].fillna(df_bureau_data['WRITE-OFF-AMT'].mean(),inplace=True)

#Dsiplaying the total null values in dataset
df_bureau_data.isna().sum()

df_train_bureau = df_bureau_data[df_bureau_data['Source']=="Train"]
df_test_bureau = df_bureau_data[df_bureau_data['Source']=="Test"]
df_train_bureau.drop("Source",axis=1,inplace=True)
df_test_bureau.drop("Source",axis=1,inplace=True)

# Data-Cleaning
def ddp_hist_func(value):
    history =  [value[i:i+3] for i in range(0,len(value),3)]
    count_no_delays = len([c for c in history if c=="000"])
    count_no_payment_history_this_month = len([c for c in history if c=="DDD"])
    count_no_payment_history_prior = len([c for c in history if c=="XXX"])
    count_payments_past_due_date = len(history)-count_no_delays-count_no_payment_history_this_month-count_no_payment_history_prior
    return count_no_delays,count_payments_past_due_date,count_no_payment_history_this_month,count_no_payment_history_prior

df_train_bureau['DPD - HIST'].fillna(",",inplace=True)
df_train_bureau['DPD_History_Months'] = df_train_bureau['DPD - HIST'].apply(ddp_hist_func)

ddp_hist =pd.DataFrame(df_train_bureau['DPD_History_Months'].tolist())
ddp_hist.columns = ['Count_No_Delays_In_Payment','Count_Payment_Past_Due_Date','History_Not_Available_Curr_Month','History_Not_Available_Prior_Month']
df_train_bureau = pd.concat([df_train_bureau,ddp_hist],axis=1)
df_train_bureau.drop('DPD_History_Months',axis=1,inplace=True)


def cur_bal_func(val):
    history = val.split(",")[:]
    history_lst = [int(item) if item!="" else 0 for item in history]
    
    max_cur_bal = max(np.array(history_lst))
    min_cur_bal = min(np.array(history_lst))
    sum_cur_bal = sum(np.array(history_lst))
    avg_cur_bal = np.mean(np.array(history_lst))
    
    sum_successive_diff = np.sum(np.diff(np.array(history_lst)))
    avg_successive_diff = sum_successive_diff/len(np.diff(np.array(history_lst)))
    
    return min_cur_bal,max_cur_bal,sum_cur_bal,avg_cur_bal,sum_successive_diff,avg_successive_diff

df_train_bureau['CUR BAL - HIST'].fillna(",",inplace=True)
df_train_bureau['CUR_Bal_History_Months'] = df_train_bureau['CUR BAL - HIST'].apply(cur_bal_func)

cur_bal_hist = pd.DataFrame(df_train_bureau['CUR_Bal_History_Months'].tolist())
cur_bal_hist.columns = ['Min_Cur_Bal','Max_Cur_Bal','Sum_Cur_Bal','Avg_Cur_Bal','Sum_Successive_Diff_CurBal','Avg_Successive_Diff_CurBal']
df_train_bureau = pd.concat([df_train_bureau, cur_bal_hist],axis=1)
df_train_bureau.drop('CUR_Bal_History_Months',axis=1,inplace=True)


def amt_overdue_func(val):
    history = val.split(",")
    history_lst = [int(item) if item!="" else 0 for item in history]
    
    max_overdue = max(np.array(history_lst))
    min_overdue = min(np.array(history_lst))
    sum_overdue = sum(np.array(history_lst))
    avg_overdue = np.mean(np.array(history_lst))
    
    sum_successive_diff = np.sum(np.diff(np.array(history_lst)))
    avg_successive_diff = sum_successive_diff/len(np.diff(np.array(history_lst)))
        
    return min_overdue,max_overdue,sum_overdue,avg_overdue,sum_successive_diff,avg_successive_diff

df_train_bureau['AMT OVERDUE - HIST'].fillna(",",inplace=True)
df_train_bureau['AMT_Overdue_History_Months'] = df_train_bureau['AMT OVERDUE - HIST'].apply(amt_overdue_func)

amt_overdue_hist =pd.DataFrame(df_train_bureau['AMT_Overdue_History_Months'].tolist())
amt_overdue_hist.columns = ['Min_Overdue','Max_Overdue','Sum_Overdue','Avg_Overdue','Sum_Successive_Diff_Overdue','Avg_Successive_Diff_Overdue']
df_train_bureau = pd.concat([df_train_bureau, amt_overdue_hist],axis=1)
df_train_bureau.drop('AMT_Overdue_History_Months',axis=1,inplace=True)


def amt_paid_func(val):
    history = val.split(",")
    history_lst = [int(float(item)) if item!="" else 0 for item in history]
    
    max_amount_paid = max(np.array(history_lst))
    min_amount_paid = min(np.array(history_lst))
    sum_amount_paid = sum(np.array(history_lst))
    avg_amount_paid = np.mean(np.array(history_lst))
    
    return min_amount_paid,max_amount_paid,sum_amount_paid,avg_amount_paid

df_train_bureau['AMT PAID - HIST'].fillna(",",inplace=True)
df_train_bureau['AMT_Paid_History_Months'] = df_train_bureau['AMT PAID - HIST'].apply(amt_paid_func)

#This is the way how you transform and rejoin a column to the dataframe in pandas
amt_paid_hist = pd.DataFrame(df_train_bureau['AMT_Paid_History_Months'].tolist())
amt_paid_hist.columns = ['Min_Amount_Paid','Max_Amount_Paid','Sum_Amount_Paid','Avg_Amount_Paid']
df_train_bureau = pd.concat([df_train_bureau,amt_paid_hist],axis=1)
df_train_bureau.drop('AMT_Paid_History_Months',axis=1,inplace=True)


def installment_amt_func(val):
    amount = val.split("/")[0]
    duration = val.split("/")[-1]
    return amount,duration

df_train_bureau['INSTALLMENT-AMT'].fillna("",inplace=True)
df_train_bureau['INSTALLMENT_HIST'] = df_train_bureau['INSTALLMENT-AMT'].apply(installment_amt_func)

installment_amt_hist = pd.DataFrame(df_train_bureau['INSTALLMENT_HIST'].tolist())
installment_amt_hist.columns = ['Installment_Amount','Installment_Frequency']
df_train_bureau = pd.concat([df_train_bureau, installment_amt_hist],axis=1)
df_train_bureau.drop(['INSTALLMENT_HIST','Installment_Frequency'],axis=1,inplace=True)

del installment_amt_hist,ddp_hist,amt_paid_hist,amt_overdue_hist,cur_bal_hist #Dropping unnecessary columns

df_train_bureau.drop(['AMT PAID - HIST','AMT OVERDUE - HIST','CUR BAL - HIST','DPD - HIST'],axis=1,inplace=True)

def ddp_hist_func(value):
    history =  [value[i:i+3] for i in range(0,len(value),3)]
    count_no_delays = len([c for c in history if c=="000"])
    count_no_payment_history_this_month = len([c for c in history if c=="DDD"])
    count_no_payment_history_prior = len([c for c in history if c=="XXX"])
    count_payments_past_due_date = len(history)-count_no_delays-count_no_payment_history_this_month-count_no_payment_history_prior
    return count_no_delays,count_payments_past_due_date,count_no_payment_history_this_month,count_no_payment_history_prior

df_test_bureau['DPD - HIST'].fillna(",",inplace=True)
df_test_bureau['DPD_History_Months'] = df_test_bureau['DPD - HIST'].apply(ddp_hist_func)

ddp_hist = pd.DataFrame(df_test_bureau['DPD_History_Months'].tolist())
ddp_hist.columns = ['Count_No_Delays_In_Payment','Count_Payment_Past_Due_Date','History_Not_Available_Curr_Month','History_Not_Available_Prior_Month']
df_test_bureau = pd.concat([df_test_bureau,ddp_hist],axis=1)
df_test_bureau.drop('DPD_History_Months',axis=1,inplace=True)


def cur_bal_func(val):
    history = val.split(",")[:-1]
    history_lst = [int(item) if item!="" else 0 for item in history]
    
    max_cur_bal = max(np.array(history_lst))
    min_cur_bal = min(np.array(history_lst))
    sum_cur_bal = sum(np.array(history_lst))
    avg_cur_bal = np.mean(np.array(history_lst))
    
    sum_successive_diff = np.sum(np.diff(np.array(history_lst)))
    avg_successive_diff = sum_successive_diff/len(np.diff(np.array(history_lst)))
    
    return min_cur_bal,max_cur_bal,sum_cur_bal,avg_cur_bal,sum_successive_diff,avg_successive_diff

df_test_bureau['CUR BAL - HIST'].fillna(",",inplace=True)
df_test_bureau['CUR_Bal_History_Months'] = df_test_bureau['CUR BAL - HIST'].apply(cur_bal_func)

cur_bal_hist = pd.DataFrame(df_test_bureau['CUR_Bal_History_Months'].tolist())
cur_bal_hist.columns = ['Min_Cur_Bal','Max_Cur_Bal','Sum_Cur_Bal','Avg_Cur_Bal','Sum_Successive_Diff_CurBal','Avg_Successive_Diff_CurBal']
df_test_bureau = pd.concat([df_test_bureau, cur_bal_hist],axis=1)
df_test_bureau.drop('CUR_Bal_History_Months',axis=1,inplace=True)


def amt_overdue_func(val):
    history = val.split(",")
    history_lst = [int(item) if item!="" else 0 for item in history]
    
    max_overdue = max(np.array(history_lst))
    min_overdue = min(np.array(history_lst))
    sum_overdue = sum(np.array(history_lst))
    avg_overdue = np.mean(np.array(history_lst))
    
    sum_successive_diff = np.sum(np.diff(np.array(history_lst)))
    avg_successive_diff = sum_successive_diff/len(np.diff(np.array(history_lst)))
        
    return min_overdue,max_overdue,sum_overdue,avg_overdue,sum_successive_diff,avg_successive_diff

df_test_bureau['AMT OVERDUE - HIST'].fillna(",",inplace=True)
df_test_bureau['AMT_Overdue_History_Months'] = df_test_bureau['AMT OVERDUE - HIST'].apply(amt_overdue_func)

amt_overdue_hist =pd.DataFrame(df_test_bureau['AMT_Overdue_History_Months'].tolist())
amt_overdue_hist.columns = ['Min_Overdue','Max_Overdue','Sum_Overdue','Avg_Overdue','Sum_Successive_Diff_Overdue','Avg_Successive_Diff_Overdue']
df_test_bureau = pd.concat([df_test_bureau, amt_overdue_hist],axis=1)
df_test_bureau.drop('AMT_Overdue_History_Months',axis=1,inplace=True)


def amt_paid_func(val):
    history = val.split(",")
    history_lst = [int(float(item)) if item!="" else 0 for item in history]
    
    max_amount_paid = max(np.array(history_lst))
    min_amount_paid = min(np.array(history_lst))
    sum_amount_paid = sum(np.array(history_lst))
    avg_amount_paid = np.mean(np.array(history_lst))
    
    
    return min_amount_paid,max_amount_paid,sum_amount_paid,avg_amount_paid

df_test_bureau['AMT PAID - HIST'].fillna(",",inplace=True)
df_test_bureau['AMT_Paid_History_Months'] = df_test_bureau['AMT PAID - HIST'].apply(amt_paid_func)

amt_paid_hist = pd.DataFrame(df_test_bureau['AMT_Paid_History_Months'].tolist())
amt_paid_hist.columns = ['Min_Amount_Paid','Max_Amount_Paid','Sum_Amount_Paid','Avg_Amount_Paid']
df_test_bureau = pd.concat([df_test_bureau, amt_paid_hist],axis=1)
df_test_bureau.drop('AMT_Paid_History_Months',axis=1,inplace=True)


def installment_amt_func(val):
    amount = val.split("/")[0]
    duration = val.split("/")[-1]
    return amount,duration

df_test_bureau['INSTALLMENT-AMT'].fillna("",inplace=True)
df_test_bureau['INSTALLMENT_HIST'] = df_test_bureau['INSTALLMENT-AMT'].apply(installment_amt_func)

installment_amt_hist = pd.DataFrame(df_test_bureau['INSTALLMENT_HIST'].tolist())
installment_amt_hist.columns = ['Installment_Amount','Installment_Frequency']
df_test_bureau = pd.concat([df_test_bureau,installment_amt_hist],axis=1)
df_test_bureau.drop(['INSTALLMENT_HIST','Installment_Frequency'],axis=1,inplace=True)


del installment_amt_hist,ddp_hist,amt_paid_hist,amt_overdue_hist,cur_bal_hist

df_test_bureau.drop(['AMT PAID - HIST','AMT OVERDUE - HIST','CUR BAL - HIST','DPD - HIST'],axis=1,inplace=True)

df_train_bureau['Source'] = "Train"
df_test_bureau['Source'] = "Test"
df_bureau_data = pd.concat([df_train_bureau, df_test_bureau],axis=0)
df_bureau_data.shape

#Display initial few rows of bureau data
df_bureau_data.head()

""" **EDA - Main Data**"""

for col in df.columns:
    if df[col].dtype == "object":
        print ("Value Counts of {} column are:\n{}".format(col,df[col].value_counts(dropna=False)))

#Creating a copy of the classification labels and converting them to numerical format using the following operations
Y_copy = Y.copy()
labels = list(set(Y.tolist()))
print(labels)

#Converting labels to numerical encodings
label_dict = {}
for i in range(len(labels)):
  label_dict[labels[i]] = i
Y_copy = Y_copy.map(label_dict)

# We can remove the following columns, same way as we did earlier
df.drop(['Area','City','BranchID','AssetID','ManufacturerID','SupplierID','City','ZiPCODE'], axis=1, inplace=True)

# Replacing some values in some columns based on their occurences, same as we did earlier
df['PaymentMode'].replace({"SI Reject":"Reject",
                           "ECS Reject":"Reject",
                           "PDC Reject":"Reject",
                           "Cheque":"PDC",
                           "PDC_E":"PDC",
                           "Escrow":"Reject",
                           "Auto Debit":"Direct Debit"}, inplace=True)

df['State'].replace({"HIMACHAL PRADESH":"Others",
                     "JHARKHAND":"Others",
                     "ASSAM":"Others",
                     "DELHI":"Others",
                     "CHANDIGARH":"Others",
                     "TAMIL NADU":"Others",
                     "DADRA AND NAGAR HAVELI":"Others"}, inplace=True)

df.drop(df[df['MaturityDAte'].isna()].index,axis=0,inplace=True)
df['SEX'].fillna("Missing",inplace=True)

#Replace missing 'AGE' values and 'MonthlyIncome' values with mean
df['AGE'].fillna(df['AGE'].mean(),inplace=True)
df['MonthlyIncome'].fillna(df['MonthlyIncome'].mean(),inplace=True)

df_bureau_data['Installment_Amount'].fillna(0,inplace=True)
df_bureau_data['Installment_Amount'] = df_bureau_data['Installment_Amount'].str.replace(",","")
df_bureau_data['Installment_Amount'] = pd.to_numeric(df_bureau_data['Installment_Amount'])

"""**Aggregate function based additional features - make a common function out of these for modular code**"""

def agg_func(df, df_bureau_data, col):
  agg_func_dict = {
      col : ['min', 'max', 'mean', 'sum']
  }

  agg_func_data = df_bureau_data.groupby('ID').agg(agg_func_dict)
  agg_func_data.columns = ['ID_' + ('_'.join(col).strip()) for col in agg_func_data.columns]
  agg_func_data.reset_index(inplace=True)
  df = df.merge(agg_func_data, on=['ID'], how='left')

  return df

df = agg_func(df, df_bureau_data, 'Count_No_Delays_In_Payment')

df = agg_func(df, df_bureau_data, 'Count_Payment_Past_Due_Date')

df = agg_func(df, df_bureau_data, 'History_Not_Available_Curr_Month')

df = agg_func(df, df_bureau_data, 'History_Not_Available_Prior_Month')

df = agg_func(df, df_bureau_data, 'Min_Cur_Bal')

df = agg_func(df, df_bureau_data, 'Max_Cur_Bal')

df = agg_func(df, df_bureau_data, 'Sum_Cur_Bal')

df = agg_func(df, df_bureau_data, 'Avg_Cur_Bal')

df = agg_func(df, df_bureau_data, 'Sum_Successive_Diff_CurBal')

df = agg_func(df, df_bureau_data, 'Avg_Successive_Diff_CurBal')

df = agg_func(df, df_bureau_data, 'Min_Amount_Paid')

df = agg_func(df, df_bureau_data, 'Max_Amount_Paid')

df = agg_func(df, df_bureau_data, 'Sum_Amount_Paid')

df = agg_func(df, df_bureau_data, 'Avg_Amount_Paid')

df = agg_func(df, df_bureau_data, 'Installment_Amount')

df = agg_func(df, df_bureau_data, 'DISBURSED-AMT/HIGH CREDIT')

df = agg_func(df, df_bureau_data, 'CURRENT-BAL')

#df['Top-up Month'].fillna(0, inplace=True)

"""**Creating Further Additional Features**"""

self_indicator_dict = df_bureau_data.groupby("ID")['SELF-INDICATOR'].sum().to_dict()
acc_type_dict = df_bureau_data.groupby("ID")['ACCT-TYPE'].nunique().to_dict()
cont_type_dict = df_bureau_data.groupby("ID")['CONTRIBUTOR-TYPE'].nunique().to_dict()

df['Count_Self_Indicator'] = df["ID"].map(self_indicator_dict)
df['Account_Type_Unique'] = df["ID"].map(acc_type_dict)
df['Unique_Contributor_Unique'] = df["ID"].map(cont_type_dict)

# Creating some more features, leveraging data of one table and fitting into another.
df['Common_ACCT-TYPE_By_ID_mode'] = df['ID'].map(df_bureau_data.groupby("ID")['ACCT-TYPE'].apply(lambda x: x.mode()[0]))
df['Common_CONTRIBUTOR-TYPE_By_ID_mode'] = df['ID'].map(df_bureau_data.groupby("ID")['CONTRIBUTOR-TYPE'].apply(lambda x: x.mode()[0]))
df['Common_OWNERSHIP-IND_By_ID_mode'] = df['ID'].map(df_bureau_data.groupby("ID")['OWNERSHIP-IND'].apply(lambda x: x.mode()[0]))
df['Common_ACCOUNT-STATUS_By_ID_mode'] = df['ID'].map(df_bureau_data.groupby("ID")['ACCOUNT-STATUS'].apply(lambda x: x.mode()[0]))

"""**Categorical Value Encoding**"""

cat_columns = []
for col in df.columns:
  if df[col].dtype == 'object':
    df[col] = np.where(df[col].isnull(), "Missing", df[col])
    cat_columns.append(col)


#Target-Oriented Mean Encoding, seems intuitive
#payment_mode_series = df.groupby('Frequency')['Top-up Month'].mean()

#Target-Oriented Mean Encoding, seems intutitive
#payment_mode_dict = payment_mode_series.to_dict()

##Target-Oriented Mean Encoding, seems intutitive
# df['Frequency_encode'] = df['Frequency'].map(payment_mode_dict)
# df.drop(['Frequency'], axis=1, inplace=True)

for col in df.columns:
  if df[col].dtype == 'object':
    df[col] = np.where(df[col].isnull(), "Missing", df[col])

df = pd.get_dummies(df, drop_first=True)


"""**Date-Time Value Encoding/Cleaning:**"""

df['DisbursalDate_Year'] = df['DisbursalDate'].dt.year
df['MaturityDAte_Quarter'] = df['MaturityDAte'].dt.quarter 
df['DisbursalMonth_Month'] = df['DisbursalDate'].dt.month  
df['DisbursalDate_Month_Start']=df['DisbursalDate'].dt.is_month_start
df['DisbursalDate_Month_End']=df['DisbursalDate'].dt.is_month_end

df['MaturityYear']=df['MaturityDAte'].dt.year
df['MaturityQuarter']=df['MaturityDAte'].dt.quarter
df['MaturityMonth']=df['MaturityDAte'].dt.month
df['MaturityDate_is_Month_Start']=df['MaturityDAte'].dt.is_month_start
df['MaturityDate_is_Month_End']=df['MaturityDAte'].dt.is_month_end

df.drop(['DisbursalDate', 'MaturityDAte'], axis=1, inplace=True)

"""**Feature Selection**"""

#Storing the reverse encodings
top_up_monthDict = {
                            "12-18 Months":0,
                            "18-24 Months":1,
                            "24-30 Months":2,
                            "30-36 Months":3,
                            "36-48 Months":4,
                            " > 48 Months":5,
                            "No Top-up Service":6
                  }
value_to_month_encoding = {value : month for (month, value) in top_up_monthDict.items()}

df.drop(['AuthDate'], axis=1, inplace=True)


#Let's study the Spearman's correlation plot among the features
corr_matrix = df.corr(method='spearman')

#Plotting the heatmap

#The following lines of code just take too much time to compute 
'''
mask = np.zeros_like(corr_matrix)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(20, 15))
    ax = sns.heatmap(corr_matrix, mask=mask, vmax=.3, square=True)
'''

#Custom function to compute features which are highly correlated. Remember, correlation doesn't imply causation.
def correlation(dataset, threshold):
  col_corr = set()
  corr_matrix = dataset.corr()
  for i in range(len(corr_matrix.columns)):
    for j in range(i):
      if (corr_matrix.iloc[i, j] > threshold):
        col_corr.add(corr_matrix.columns[i])
  return col_corr

high_col_cors = set()
high_col_cors = list(correlation(df, 0.8))

#Drop the highly correlated columns
df.drop(high_col_cors, axis=1, inplace=True)



#Oops one column is still left to be encoded
# disbursal_values={}
# for idx, value in enumerate(df['DisbursalDate_Month_End'].unique()):
#   disbursal_values[value] = idx

#Label and drop the categorical column
# df['DisbursalDate_Month_End_Labelled'] = df['DisbursalDate_Month_End'].map(disbursal_values)
# df.drop(['DisbursalDate_Month_End'], axis=1, inplace=True)

#Again splitting into train and test sets
df_train_1 = df[ : len(df_train)]
df_test_1 = df[len(df_train) : ] 
X = df_train_1
X_test = df_test_1

#Train-Test split, performing simple hold-out validation initially, we 
X_train, X_valid, y_train, y_valid = train_test_split(X, Y_copy, test_size=0.3, random_state=seed_value, shuffle=True, stratify=Y_copy)

#Printing the shape of train and validation sets
print(X_train.shape)
print(X_valid.shape)
print(y_train.shape)
print(y_valid.shape)

# df_train_1.drop(['Source'], axis=1, inplace=True)
# df_test_1.drop(['Source'], axis=1, inplace=True)

#Replace all NaN values with 0 cause there are a lot may data for the model to learn and adapt
X_train.isna().sum()

#Further drop some columns and impute the rest of them
X_train.drop(['ID_Installment_Amount_min', 'ID_Installment_Amount_max', 'ID_Installment_Amount_mean'], axis=1, inplace=True)
X_test.drop(['ID_Installment_Amount_min', 'ID_Installment_Amount_max', 'ID_Installment_Amount_mean'], axis=1, inplace=True)

#Dropping columns from X_valid as well
X_valid.drop(['ID_Installment_Amount_min', 'ID_Installment_Amount_max', 'ID_Installment_Amount_mean'], axis=1, inplace=True)

X.drop(['ID'], axis=1, inplace=True)

len(X_test.columns)

#Impute rest of the columns with 0
X_train.fillna(0, axis=1, inplace=True)
X_valid.fillna(0, axis=1, inplace=True)
X_test.fillna(0, axis=1, inplace=True)

#Modelling - (Hyperparameter Tuning + GridSearchCV + KFold)

#Initialising the classifiers
lr = LogisticRegression(random_state=seed_value)
rmr = RandomForestClassifier(random_state=seed_value)
adbr = AdaBoostClassifier(random_state=seed_value)
xgbr = XGBRFClassifier(random_state=seed_value)

#Parameters for each of the models

#Logistic Regression Classifier
params_grid_lr = {
             "penalty" : ['l1', 'l2', 'elasticnet', 'none'],
             "C"       :  [0.001,0.01,0.1,1,10,100,1000],
             "fit_intercept" : [True, False],
             "solver" : ['newton-cg', 'sag', 'saga']
}

#Random Forest Regressor
params_grid_rmr = {
              "n_estimators" : [int(i) for i in np.linspace(100, 500, 10)],
              "min_samples_split": [10, 20, 40],
              "criterion" : ['gini', 'entropy'],
              "max_depth": [2, 6, 8],
              "min_samples_leaf": [20, 40, 100],
              "max_leaf_nodes": [5, 20, 100],
              "bootstrap" : [True, False],
              "max_features" : ['auto', 'sqrt', 'log2']
}

#XGBoost Regressor
params_grid_xgbr = {
             "learning_rate" : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40], 
             "max_depth" : [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 
             "min_child_weight" : [1, 3, 5, 7], 
             "gamma" : [0.0, 0.1, 0.2, 0.3, 0.4],
             "colsample_bytree" : [0.3, 0.5, 0.7], 
             "colsample_bylevel" : [0.2, 0.4, 0.6],
             "subsample" : [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
}

#AdaBoost Regressor
params_grid_adbr = {
            "learning_rate" : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
            "n_estimators" : [int(i) for i in np.linspace(100, 500, 10)], 
}

#Using RandomizedSearch CV for training with cross validation, since its a very large dataset. Since its a large dataset, the labels will be preserved.
def fine_tune_params(classifier, parameters, X_train, y_train):
  rm = RandomizedSearchCV(estimator=classifier, param_distributions=parameters, n_iter=8, n_jobs=-1, cv=9, random_state=seed_value)
  search = rm.fit(X_train, y_train)
  bs = search.best_score_
  bp = search.best_params_
  return bs, bp

#Logistic Regression

acc_scores, roc_auc_scores = [], []

#Logistic Regression Classifier
#bs_logistic, bp_logistic = fine_tune_params(lr, params_grid_lr, X_train, y_train)
##Grid SearchCV or Randomized SearchCV taking too long to respond, Hence we will handpick some parameters as mentioned in parameter grid
lr = LogisticRegression(fit_intercept=True, solver='saga', random_state=seed_value, C=0.01)
lr.fit(X_train, y_train)

#Classification metric evaluations
pred_lr = lr.predict(X_valid)
print(accuracy_score(y_valid, pred_lr))
acc_scores.append(accuracy_score(y_valid, pred_lr))

print(confusion_matrix(y_valid, pred_lr))
print(classification_report(y_valid, pred_lr))

# y_true = np.argmax(y_valid, axis=0)
# y_valid.values
# y_valid = y_valid.reshape(y_valid)
# pred_lr_proba = lr.predict_proba(X_valid)

# # roc curve for classes
# fpr = {}
# tpr = {}
# thresh ={}
# roc_auc_score(y_valid, pred_lr_proba, multi_class='ovr')

#Random Forest Classifier
rmr = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=20, 
                             min_samples_leaf=20, bootstrap=True,
                             random_state=seed_value)
rmr.fit(X_train, y_train)

#Classification metric evaluations
pred_rmr = rmr.predict(X_valid)
print(accuracy_score(y_valid, pred_rmr))
acc_scores.append(accuracy_score(y_valid, pred_rmr))
acc_scores.append(accuracy_score(y_valid, pred_rmr))

print(confusion_matrix(y_valid, pred_rmr))
print(classification_report(y_valid, pred_rmr))


#XGBRF Classifier
xgbr = XGBRFClassifier(learning_rate=0.01, min_child_weight=5, max_depth=10, 
                       gamma=0.05, colsample_bytree=0.6, colsample_bylevel=0.6,
                       subsample=0.9)
xgbr.fit(X_train, y_train)

#Classification metric evaluations
pred_xgbr = xgbr.predict(X_valid)
print(accuracy_score(y_valid, pred_xgbr))
acc_scores.append(accuracy_score(y_valid, pred_xgbr))

print(confusion_matrix(y_valid, pred_xgbr))
print(classification_report(y_valid, pred_xgbr))

#LGBMClassifier + Stratified K-Fold(since its an imbalanced dataset)
X.drop(['ID_Installment_Amount_min', 'ID_Installment_Amount_max', 'ID_Installment_Amount_mean'], axis=1, inplace=True)

#Stratified K-Fold Cross Validation
#kf = StratifiedKFold(n_splits=10, random_state=seed_value, shuffle=True)

# for train_index,test_index in kf.split(X,y):
#      print('\n{} of kfold {}'.format(i,kf.n_splits))
#      xtr,xvl = X.loc[train_index],X.loc[test_index]
#      ytr,yvl = y[train_index],y[test_index]
#      model = GridSearchCV(XGBClassifier(), param_grid, cv=10, scoring= 'f1', iid=True)
#      model.fit(xtr, ytr)
#      print (model.best_params_)
#      pred=model.predict(xvl)
#      print('accuracy_score',accuracy_score(yvl,pred))
#      i+=1

#Creating a copy of X as well
X_copy = X.copy()

#Dropping columns yet again, will perform discretization sometime later
cols_to_drop = []
for col in X_copy.columns:
   if X_copy[col].nunique() < 30:
      cols_to_drop.append(col)

X_copy.drop(cols_to_drop, axis=1, inplace=True)

X_test.drop(cols_to_drop, axis=1, inplace=True)

len(X_test.columns)

X_test.drop(['ID'], axis=1, inplace=True)

# Modelling with Stratified K Fold cross validation, there is surely some issue with the final predictions
f1 = []
final_preds = []
folds = StratifiedKFold(n_splits=5)
for train_index, test_index in folds.split(X_copy, Y_copy.values):

    # creating training and validation datasets
    X_Train, X_Test = X_copy.iloc[train_index], X_copy.iloc[test_index]
    y_Train, y_Test = Y_copy.iloc[train_index], Y_copy.iloc[test_index]
    
    # building a classifier
    clf = LGBMClassifier(n_estimators=750,
                             learning_rate=0.1,
                             objective="multiclass",
                             boosting_type="gbdt",
                             subsample=0.9,
                             colsample_bytree=0.6,
                             num_class=7,
                             max_depth=12,
                             n_jobs=-1,
                             reg_alpha=2,
                             num_leaves=100,
                             class_weight='balanced')
    
    # fitting the classifier to the train set
    clf.fit(X_Train,
            y_Train,
            eval_set=[(X_Train, y_Train),(X_Test, y_Test)],
            early_stopping_rounds=50,
            verbose=50)
    
    # predicting on the validation set
    preds = clf.predict(X_Test)
    score = f1_score(preds, y_Test, average='macro')
    print ("F1 Score:",score)
    print ("------------------------------------------------")
    f1.append(score)
    
    # Predicting on the test set, final_preds will be an array of 5 sets of predictions, one for each fold
    pred = clf.predict(X_test, num_iteration=None)
    final_preds.append(pred)
    
print ("------------------------------------------")
print ("Mean F1 Score of 5 Folds:", np.mean(np.array(f1)))

"""**Test Dataset Predictions**"""

y_valid.values

ser = pd.Series(clf.feature_importances_, X_copy.columns).sort_values()
ax = ser.plot(kind='bar', figsize=(20, 18))
plt.xticks(size=15)


# Commented out IPython magic to ensure Python compatibility.
# Dumping ML models to file

#Saving Logistic Regression
joblib.dump(lr, 'B:/ML-NLP/Projects/lTFsdeploy/lTFs-finance/saved models/lr.pkl')
joblib.dump(lr, 'B:/ML-NLP/Projects/lTFsdeploy/lTFs-finance/saved models/lr.sav')

#Saving Random Forest Classifier
joblib.dump(rmr, 'B:/ML-NLP/Projects/lTFsdeploy/lTFs-finance/saved models/rmr.pkl')
joblib.dump(rmr, 'B:/ML-NLP/Projects/lTFsdeploy/lTFs-finance/saved models/rmr.sav')

#Saving XGBoost classifier
joblib.dump(xgbr, 'B:/ML-NLP/Projects/lTFsdeploy/lTFs-finance/saved models/xgbr.pkl')
joblib.dump(xgbr, 'B:/ML-NLP/Projects/lTFsdeploy/lTFs-finance/saved models/xgbr.sav')

#Saving LGBM classifier
joblib.dump(clf, 'B:/ML-NLP/Projects/lTFsdeploy/lTFs-finance/saved models/lgbm.pkl')
joblib.dump(clf, 'B:/ML-NLP/Projects/lTFsdeploy/lTFs-finance/saved models/lgbm.sav')