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

ticker = 'SYK'

# Get path of set
if os.name == 'posix':
    input_dir = r'/home/jskrable/code/cs677/datasets'
else:
    input_dir = r'C:\Users\jskrable\code\cs677\datasets'
output_file = os.path.join(input_dir, ticker + '.csv')

# Read file into pandas dataframe
df = pd.read_csv(output_file)


# Helper functions
# Calculate single week's net return
def net(week):
    week = week.to_numpy()
    return (week[-1] - week[0])/week[0]


# Calculate bool for week's volatility
def vol(week):
    w_std = week.std()
    # Consider volatile if weekly standard dev > overall mean weekly std dev
    return True if w_std > std else False


# Perform group functions
def grp_vol(grp):
    grp['Vol'] = vol(grp['Return'])
    return grp


def grp_net(grp):
    grp['Net'] = net(grp['Adj Close'])
    return grp


def grp_mean_return(grp):
    grp['Avg Return'] = grp['Return'].mean()
    return grp


# Apply group calculations and return to parent df
def group_apply(data, group):
    data['Net'] = group.apply(grp_net)['Net']
    data['Vol'] = group.apply(grp_vol)['Vol']
    data['Avg Return'] = group.apply(grp_mean_return)['Avg Return']
    data['Score'] = data.apply(
        (lambda x: 1 if x['Vol'] == False and x['Net'] > 0 else 0), axis=1)


# Get weekly summary dataframe
df['Date'] = pd.to_datetime(df['Date']) - pd.to_timedelta(7, unit='d')
# Get weekly avg std dev for comparison
std = df.groupby([pd.Grouper(key='Date', freq='W')])['Return'].std().mean()
group = df.groupby([pd.Grouper(key='Date', freq='W')])
group_apply(df,group)
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

# Try ks, get error rate
error_rate = {}
Ks = [x for x in range(1, 22, 2)]
for k in Ks:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    pred_k = knn.predict(x_test)
    error_rate.update({k: np.mean(pred_k != y_test)})


# Plot error rate
def error_plot(rate):
    x = [k for k in rate.keys()]
    y = [v for v in rate.values()]
    plt.plot(x, y, 'o:')
    plt.xticks(x)
    plt.ylabel('Error Rate')
    plt.xlabel('k-value')
    plt.title('Error Rate vs. k: SYK Weekly Return Classifier')
    plt.show()

error_plot(error_rate)