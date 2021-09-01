#Loading Dependencies
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.adapt import MLkNN
from skmultilearn.ensemble import MajorityVotingClassifier
from skmultilearn.cluster import FixedLabelSpaceClusterer

#Reading in the data
data = pd.read_csv("heartfailuredata.csv")

print(data.shape)

#Splitting Up the Data
cols1 = [21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,46]
medical_history = data[data.columns[cols1]]
medical_history

cols2 = [1,2,3,4,5,6,7,8,166]
demographic = data[data.columns[cols2]]
demographic

cols3 = [9,10,11,12,13,14,15,16,17,18,19,20,39,40,41,42,43,44,164]
baseline_clinical_characteristics = data[data.columns[cols3]]
baseline_clinical_characteristics

cols4 = [47,48,49,50,51,52,53]
echo = data[data.columns[cols4]]
echo

cols5 = [160,161,162,163,133,134,135,136,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,66,67,68,69,93,100,101,102,103,104,105,106,107,149,150,151,152,153,157,141,158,159,145,45]
blood_work = data[data.columns[cols5]]
blood_work

cols6 = [121,122,123,124,126,114,115,116,117,118,108,109,110,111,112,92,65,94,95,96,97,98,99,137,138,139,140,146,132,119,120,124,125,127,128,129,130,131,156,148,142,143,144,147,154,155]
metabolomics = data[data.columns[cols6]]
metabolomics

cols7 = [165,54,55,56,57,58,59,60,61,62,63,64]
outcomes = data[data.columns[cols7]]
outcomes



###Data Cleaning###



#One Hot Encoding for Demographics Data
demographics_cleaned = demographic
categorical_columns = ['DestinationDischarge', 'admission.ward', 'admission.way', 'occupation', 'discharge.department','gender', 'ageCat']

for col in categorical_columns:
    col_ohe = pd.get_dummies(demographic[col], prefix=col)
    demographics_cleaned = pd.concat((demographics_cleaned, col_ohe), axis=1).drop(col, axis=1)

demographics_cleaned.to_csv(r'cleaned data/demographics_cleaned.csv', index=False)


#Imputing Missing Values for Baseline Clinical Chracteristics 
print((baseline_clinical_characteristics == 0).sum())
baseline_clinical_characteristics_cleaned = baseline_clinical_characteristics.replace(0, np.nan)
baseline_clinical_characteristics_cleaned.fillna(baseline_clinical_characteristics.mean(), inplace=True)
print(baseline_clinical_characteristics_cleaned.isnull().sum())
print(baseline_clinical_characteristics_cleaned.shape)
baseline_clinical_characteristics_cleaned.to_csv(r'cleaned data/clinical_cleaned.csv', index=False)


#One Hot Encoding for Baseline Clinical Data
baseline_clinical_characteristics_cleaned_ohe = baseline_clinical_characteristics_cleaned
categorical_columns = ['type.of.heart.failure', 'NYHA.cardiac.function.classification', 'Killip.grade', 'consciousness' ,'respiratory.support.', 'oxygen.inhalation']

for col in categorical_columns:
    col_ohe = pd.get_dummies(baseline_clinical_characteristics_cleaned[col], prefix=col)
    baseline_clinical_characteristics_cleaned_ohe = pd.concat((baseline_clinical_characteristics_cleaned_ohe, col_ohe), axis=1).drop(col, axis=1)

baseline_clinical_characteristics_cleaned_ohe.to_csv(r'cleaned data/baseline_clinical_characteristics_cleaned_ohe.csv', index=False)


#Cleaning and One Hot Encoding for Outcome during hospitilization  
cols8 = [55,56,57,58,59,60,63]
outcomes = data[data.columns[cols8]]
print(outcomes.shape)
print((outcomes == 0).sum())

outcome_during_hositilization = data[data.columns[54]]
outcome_during_hositilization_ohe = pd.get_dummies(outcome_during_hositilization)
outcomes_cleaned = pd.concat([outcomes, outcome_during_hositilization_ohe], axis = 1)
outcomes_cleaned.rename(columns = {"Alive": "outcome.during.hospitilization.alive" , "Dead": "outcome.during.hospitilization.dead" , "DischargeAgainstOrder": "outcome.during.hospitilization.DischargeAgainstOrder" }, inplace = True)
outcomes_cleaned.fillna(0, inplace = True)
outcomes_cleaned.to_csv(r'cleaned data/outcomes_cleaned.csv', index=False)


#Cleaning Echo Data
echo_cleaned = echo.replace("NA", np.nan)
echo_cleaned.fillna(echo.mean(), inplace = True)
echo_cleaned
print(echo_cleaned.isnull().sum())
print(echo.shape)
echo_cleaned.to_csv(r'cleaned data/echo_cleaned.csv', index = False)


#Cleaning Medical History Data
medical_history_ohe = medical_history
categorical_columns = ['CCI.score', 'type.II.respiratory.failure']

for col in categorical_columns:
    col_ohe = pd.get_dummies(medical_history[col], prefix=col)
    medical_history_ohe = pd.concat((medical_history_ohe, col_ohe), axis=1).drop(col, axis=1)

medical_history_cleaned = medical_history_ohe.replace(0.0,0)
medical_history_cleaned.fillna(0, inplace = True)
medical_history_cleaned.isnull().sum()
medical_history_cleaned.to_csv(r'cleaned data/medical_history_cleaned.csv', index=False)


#Cleaning Blood Work Data
blood_work_cleaned = blood_work.replace(r'\s+( +\.)|#',np.nan,regex=True).replace('',np.nan)
print(blood_work_cleaned.isnull().sum())
blood_work_cleaned.fillna(blood_work_cleaned.mean(), inplace = True)
blood_work_cleaned.to_csv(r'cleaned data/blood_work_cleaned.csv', index=False)


#Cleaning Metabalomics Data
metabolomics.to_csv(r'cleaned data/metabolomics_ugly.csv', index = False)
print(metabolomics.isnull().sum())

cols9 = [122,123,126,114,115,116,117,118,108,109,110,112,92,65,94,95,96,97,98,99,146,119,120,124,125,127,128,129,130,131,156,148,142,143,144,147,154,155]
metabolomics_removed = data[data.columns[cols9]]
metabolomics_removed
print(metabolomics_removed.isnull().sum())

metabolomics_cleaned = metabolomics_removed.replace('', np.nan)
metabolomics_cleaned.fillna(metabolomics_cleaned.mean(), inplace = True)
metabolomics_cleaned.to_csv(r'cleaned data/metabolomics_cleaned.csv', index = False)



#####Giving ID to Metabalomics Data#####



#####                             #######


#Combining Data
data_frames = [demographics_cleaned, baseline_clinical_characteristics_cleaned_ohe, echo_cleaned, medical_history_cleaned,  blood_work_cleaned, metabolomics_cleaned, outcomes_cleaned]
cleaned_heart_disease_data = pd.concat(data_frames, axis = 1)
cleaned_heart_disease_data.to_csv(r'cleaned data/cleaned_heart_disease_data.csv', index = False)

#Scaling Data
cleaned_heart_disease_data = cleaned_heart_disease_data.drop(['inpatient.number'],axis = 1)
scaler = MinMaxScaler()
scaled_cleaned_heart_disease_data = scaler.fit_transform(cleaned_heart_disease_data)
scaled_cleaned_heart_disease_data_df = pd.DataFrame(scaled_cleaned_heart_disease_data, columns = cleaned_heart_disease_data.columns)

#Outputting Scaled Data
patient_id = cleaned_heart_disease_data["inpatient.number"]
data_frames1 = [patient_id, scaled_cleaned_heart_disease_data_df]
scaled_cleaned_heart_disease_data_df1 = pd.concat(data_frames1, axis = 1)
scaled_cleaned_heart_disease_data_df1.to_csv(r'cleaned data/scaled data/scaled_cleaned_heart_disease_data.csv', index=False)

#EDA
print(scaled_cleaned_heart_disease_data_df1.shape)
print(scaled_cleaned_heart_disease_data_df1.isnull().sum())


#corr = cleaned_heart_disease_data.corr
#fig, ax = plt.subplots(figsize=(7,5))
#sns.heatmap(corr, square = True)