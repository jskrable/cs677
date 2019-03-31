#!/usr/bin/env python3
# coding: utf-8
"""
naive_bayes.py
03-31-19
jack skrable
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mode
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

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

# Extract data for classification
print('Preprocessing data...')
# Fill NaNs
weekly = weekly.fillna(0)
weekly = weekly.reset_index()
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

def split_set(data):
    x = data[['Mean','Std']].values
    y = data['Score'].values
    return x, y


def nb_gaussian(train, test):
    print('Classifying with Naive Bayes...')
    train_x, train_y = split_set(train)
    test_x, test_y = split_set(test)

    nbc = GaussianNB().fit(train_x,train_y)
    predictions = nbc.predict(test_x)

    errors = 0
    for i, y in enumerate(predictions):
        if y != test_y[i]:
            errors += 1

    return errors/test_y.shape[0]


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

print('Classifying with kNN...')
# Try all k values
for k in Ks:
    # With each distance calculation
    for o in Os:
        key = '-'.join([str(k),str(o)])
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


manhattan = min([v for k,v in error_rate.items() if k.split('-')[1] == '1'])
minikowski = min([v for k,v in error_rate.items() if k.split('-')[1] == '1.5'])
euclidean = min([v for k,v in error_rate.items() if k.split('-')[1] == '2'])
naive_bayes = nb_gaussian(train,test)

print('Q1: Naive Bayes vs. kNN-------------------------------------------------')
print('Algorithm                                 Error Rate')
print('Naive Bayes                              ',naive_bayes)
print('kNN (manhattan)                          ',manhattan)
print('kNN (minikowski)                         ',minikowski)
print('kNN (euclidean)                          ',euclidean)
