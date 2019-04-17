#!/usr/bin/env python3
# coding: utf-8
"""
svm.py
04-17-19
jack skrable
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler

ticker = 'SYK'
label = True

# Get path of set
if os.name == 'posix':
    input_dir = r'/home/jskrable/code/cs677/datasets'
else:
    input_dir = r'C:\Users\jskrable\code\cs677\datasets'
output_file = os.path.join(input_dir, ticker + '_labelled.csv')

# Read labelled file into pandas dataframe
print('Reading data file...')
df = pd.read_csv(output_file)

# Get weekly summary dataframe
print('Applying group functions...')
df['Date'] = pd.to_datetime(df['Date']) - pd.to_timedelta(7, unit='d')
group = df.groupby([pd.Grouper(key='Date', freq='W')])
weekly = group['Return'].mean().to_frame('Mean')
weekly['Std'] = group['Return'].std()
weekly['Score'] = group['Score'].first()

# Fill NaNs
weekly = weekly.fillna(0)

# Split to training and testing data
weekly = weekly.reset_index()
weekly['Year'] = weekly['Date'].dt.year
test = weekly.loc[weekly['Year'].isin([2018])]
train = weekly.loc[weekly['Year'].isin([2017])]

# Extract data for classification
print('Preprocessing data...')
x = weekly[['Mean', 'Std']].values
x_train = train[['Mean', 'Std']].values
y_train = train['Score'].values
x_test = test[['Mean', 'Std']].values
y_test = test['Score'].values

# Scale data
scaler = StandardScaler()
scaler.fit(x)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

def run_svm(kind):
    # Train SVM classifier
    print('Fitting',kind,'SVM...')
    if kind == 'poly':
        svm_classifier = svm.SVC(kernel=kind, degree=2, gamma='auto')
    else:
        svm_classifier = svm.SVC(kernel=kind, gamma='auto')
    svm_classifier.fit(x_train,y_train)

    # Test classifier
    print('Testing',kind,'SVM classifier...')
    y_pred = svm_classifier.predict(x_test)
    error_rate = np.mean(y_pred != y_test)

    return np.round(1 - error_rate, 4) * 100

linear = run_svm('linear')
gaussian = run_svm('rbf')
poly = run_svm('poly')

print('Complete---------------------')
print('Linear Accuracy:',linear,'%')
print('Gaussian Accuracy:',gaussian,'%')
print('Polynomial Accuracy:',poly,'%')
