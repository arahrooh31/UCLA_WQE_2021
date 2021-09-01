#Loading Dependencies
import pandas as pd
import numpy as np

X = pd.read_csv('cleaned data/scaled data/scaled_cleaned_heart_disease_data.csv', usecols = range(1,188), header = 0)
Y = pd.read_csv('cleaned data/scaled data/scaled_cleaned_heart_disease_data.csv', usecols = range(188,198), header = 0)


#myocardial_infarction is col 70
#Type of heart failure is 44, 45, 46