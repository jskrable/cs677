#!/usr/bin/env python3
# coding: utf-8
"""
regression_return_estimates.py
03-31-19
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


# Function to predict return of next day using linear regression
def predict_return(y, w):

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
        if prediction > y[-1]:
            return 1
        else:
            return -1


print('Q2: Rolling Regression Predicted Returns---------------------')
for w in [10,20,30]:
    rate = df['Return'].rolling(window=(w+1), min_periods=1).apply(lambda x: predict_return(x, (w+1)), raw=True)

    print('W:', w, 'Rate:', rate.sum())
