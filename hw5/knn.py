#!/usr/bin/env python3
# coding: utf-8
"""
knn.py
02-25-19
jack skrable
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import week_classifier as wk

ticker = 'SYK'

# Get path of set
if os.name == 'posix':
    input_dir = r'/home/jskrable/code/cs677/datasets'
else:
    input_dir = r'C:\Users\jskrable\code\cs677\datasets'
output_file = os.path.join(input_dir, ticker + '.csv')

# Read file into pandas dataframe
df = pd.read_csv(output_file)

# Get weekly summary dataframe
df['Date'] = pd.to_datetime(df['Date']) - pd.to_timedelta(7, unit='d')
group = df.groupby([pd.Grouper(key='Date', freq='W')])
wk.group_apply(df,group)
weekly = group['Return'].mean().to_frame('Mean')
weekly['Std'] = group['Return'].std()
weekly['Score'] = group['Score'].first()

# Fill NaNs
weekly = weekly.fillna(0)

# Split to training and testing data
weekly = weekly.reset_index()
weekly['Year'] = weekly['Date'].dt.year
test = weekly.loc[weekly['Year'].isin([2017])]
train = weekly.loc[weekly['Year'].isin([2018])]

# Extract data for classification
x_train = train[['Mean','Std']].values
y_train = train[['Score']].values
x_test = test[['Mean','Std']].values
y_test = test[['Score']].values


# Scale data
scaler = StandardScaler()
scaler.fit(x_train)
x = scaler.transform(x)

error_rate = []
for k in range(1,21,2):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    pred_k = knn.predict(x_test)
    error_rate.append(np.mean(pred_k != y_test))