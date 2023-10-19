import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#importing the SharkTank dataset
df = pd.read_csv('SharkTank.csv')

#checking the head of the dataset
df.head()

#checking if there are any null values
df.isnull().sum()

df.info()

#use KNN imputor to fill missing values in Total Deal Amount and Total D
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df['Total Deal Amount'] = imputer.fit_transform(df[['Total Deal Amount']])
df['Total Deal Equity'] = imputer.fit_transform(df[['Total Deal Equity']])
df['Total Deal Amount'] = df['Total Deal Amount'].astype(int)
df['Total Deal Equity'] = df['Total Deal Equity'].astype(int)

df.info()

#dropping columns pitchers city, season number, season start, season end, episode number, pitch number, original air date, Startup Name, Pitchers Gender, Pitchers State, Entrepreneur Names
df.drop(['Season Number', 'Season Start', 'Season End', 'Episode Number' , 'Pitch Number' , 'Original Air Date' , 'Startup Name', 'Pitchers Gender' , 'Pitchers City', 'Pitchers State', 'Entrepreneur Names'], axis=1, inplace=True)
df.drop([10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29], axis=1, inplace=True)

df.info()

#filling multiple empty column values with zero
df['Total Deal Amount'].fillna(0, inplace=True)
