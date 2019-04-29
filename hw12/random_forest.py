#!/usr/bin/env python3
# coding: utf-8
"""
random_forest.py
04-29-19
jack skrable
"""

import os
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# custom script import
import classifiers

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
y = weekly[['Score']].values
x_train = train[['Mean', 'Std']].values
y_train = train['Score'].values
x_test = test[['Mean', 'Std']].values
y_test = test['Score'].values

# Scale data
scaler = StandardScaler()
scaler.fit(x)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


def random_forest(n, d):
    clf = RandomForestClassifier(n_estimators=n, max_depth=d)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return np.mean(y_pred != y_test)


def plot_errors(errors):
    print('Plotting...')
    sns.lineplot(list(errors.keys()), list(errors.values()), marker='o')
    plt.xlabel('N-D value')
    plt.ylabel('Error Rate')
    plt.show()


def simulate_rfcs():
    print('Trying different values for N and D...')
    errors = {'-'.join([str(n), str(d)]): random_forest(n, d)
              for n in range(1, 11) for d in range(1, 6)}
    plot_errors(errors)


def test_classifiers():
    
    knn = classifiers.k_nearest_neighbor(x_train, y_train, x_test, y_test, 9)
    dtc = classifiers.decision_tree(x_train, y_train, x_test, y_test)
    lgr = classifiers.logistic_regression(x_train, y_train, x_test, y_test)
    svm = classifiers.support_vector_machine(x_train, y_train, x_test, y_test, 'poly')
    print('Classifying with random forest...')
    rfc = random_forest(10, 2)
    nvb = classifiers.naive_bayes(x_train, y_train, x_test, y_test)

    print('Classifer------------Params------Error Rate--------------------')
    print('kNN                  k=9         Error =',knn)
    print('Decision Tree        Entropy     Error =',dtc)
    print('Logistic Regression  lbfgs       Error =',lgr)
    print('Support Vector       poly        Error =',svm)
    print('Random Forest        n=10 d=2    Error =',rfc)
    print('Naive Bayes          gaussian    Error =',nvb)

    print('Plotting...')
    rates = {'knn':knn, 'dtc':dtc, 'lgr':lgr, 'svm':svm, 'rfc':rfc, 'nvb':nvb}
    sns.barplot(list(rates.keys()), list(rates.values()))
    plt.xlabel('classifier')
    plt.ylabel('error rate')
    plt.show()
    


# MAIN
##########################################################################
print('Q1---------------------------------------------------------------')
simulate_rfcs()

print('Q2---------------------------------------------------------------')
test_classifiers()
