#!/usr/bin/env python3
# coding: utf-8
"""
custom_knn.py
02-25-19
jack skrable
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mode
from sklearn.preprocessing import StandardScaler

ticker = 'SYK'

# Get path of set
if os.name == 'posix':
    input_dir = r'/home/jskrable/code/cs677/datasets'
else:
    input_dir = r'C:\Users\jskrable\code\cs677\datasets'
output_file = os.path.join(input_dir, ticker + '.csv')

# Read file into pandas dataframe
print('Reading data file...')
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
print('Applying group functions...')
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
weekly = weekly.reset_index()

# Extract data for classification
print('Preprocessing data...')
x = weekly[['Mean', 'Std']].values

# Scale data
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

# Test and training sets
weekly['Mean'], weekly['Std'] = np.hsplit(scaler.transform(x),2)
weekly = weekly[['Mean', 'Std', 'Score']]
train = weekly[int(len(weekly)/2):]
test = weekly[:int(len(weekly)/2)]


# Custom knn, takes in X and returns predicted Y
def custom_knn(train, to_predict, o, k):

    # Init list for distances
    distances = []
    # Loop thru all training values
    for row in train.values:
        # Measure distance to prediction point
        dist = np.linalg.norm((row[:-1] - to_predict), ord=o)
        # Add distance and label to list
        distances.append([dist, row[-1]])    

    # Get label for k closest neighbors
    results = [dist[1] for dist in sorted(distances)[:k]]
    # Return the highest prob. label
    return mode(results)


# Try ks, get error rate
error_rate = {}
Ks = [x for x in range(1, 22, 2)]
# List of ords to try
# Manhattan, minikowski, and euclidean 
Os = [1,1.5,2]
size = len(test)

# Try all k values
for k in Ks:
    # With each distance calculation
    for o in Os:
        key = '-'.join([str(k),str(o)])
        print('Classifying with k:', k, 'ord:', o)
        # Init bad prediction counter
        error = 0
        # Loop thru test rows
        for row in test.values:
            # Make a prediction
            pred = custom_knn(train, row[:-1], o, k)
            if pred != row[-1]:
                # Note the error
                error += 1 
        error_rate.update({key: (error/size)})


# Plot error rate
def error_plot(rate):
    # Unique k values
    x = sorted([int(x) for x in(set([k.split('-')[0] for k in rate.keys()]))])
    # Manhattan
    y1 = [v for k,v in error_rate.items() if k.split('-')[1] == '1']
    # Minkowski
    y2 = [v for k,v in error_rate.items() if k.split('-')[1] == '1.5']
    # Euclidean
    y3 = [v for k,v in error_rate.items() if k.split('-')[1] == '2']
    plt.plot(x, y1, 'o:', label='manhattan')
    plt.plot(x, y2, 'o:', label='minkowski')
    plt.plot(x, y3, 'o:', label='euclidean')
    plt.legend()
    plt.xticks(x)
    plt.ylabel('Error Rate')
    plt.xlabel('k-value')
    plt.title('Error Rate vs. k: SYK Weekly Return Classifier')
    plt.show()

print('Plotting results...')
error_plot(error_rate)