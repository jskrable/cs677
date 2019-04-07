#!/usr/bin/env python3
# coding: utf-8
"""
custom_regression.py
04-07-19
jack skrable
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ticker = 'SYK'

# Get path of set
if os.name == 'posix':
    input_dir = r'/home/jskrable/code/cs677/datasets'
else:
    input_dir = r'C:\Users\jskrable\code\cs677\datasets'
output_file = os.path.join(input_dir, ticker + '_labelled.csv')

# Read labelled file into pandas dataframe
print('Reading data file...')
df = pd.read_csv(output_file)


def custom(y, w):
    if y.shape[0] < w:
        return 0
    else:    
        x = np.array([x for x in range(w - 1)])
        n = np.size(x)
        mx, my = np.mean(x), np.mean(y[:-1])
        ss_xy = np.sum(x*y[:-1]) - (n * my * mx)
        ss_xx = np.sum(x*x) - (n * mx * mx)
        slope = ss_xy / ss_xx
        intercept = my - (slope * mx)
        prediction = (slope * w) + intercept
        if (prediction > 0 and y[-1] > 0) or (prediction < 0 and y[-1] < 0):
            return 1
        else:
            return 0


# Function to predict return of next day using linear regression
def polyfit(y, w):

    # Do not attempt if there are not enough preceding days
    if y.shape[0] < w:
        return 0
    else:
        # Get x values
        x = np.array([x for x in range(w - 1)])
        # Polyfit weights for x and y
        weights = np.polyfit(x,y[:-1],1)
        # Model creation
        model = np.poly1d(weights)
        # Predict
        prediction = model(w+1)

        # Return rate
        if (prediction > 0 and y[-1] > 0) or (prediction < 0 and y[-1] < 0):
            return 1
        else:
            return 0


def simulate(algorithm):
    
    for w in [10,20,30]:
        rate = df['Return'].rolling(window=(w+1), min_periods=1).apply(lambda x: algorithm(x, (w+1)), raw=True)
        success_rate = np.round((rate.sum()/rate.shape[0]), 4)*100
        print('W:', w, '          Success Rate:', success_rate, '%')


# MAIN
################################################################################
print('Q1: Rolling Regression Predicted Returns-------------------------------')
print('Custom Regression Formula')
simulate(custom)
print('NumPy Polyfit')
simulate(polyfit)