#!/usr/bin/env python3
# coding: utf-8
"""
gradient_descent.py
04-07-19
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


def custom_gradient_descent(y, w, L):
    if y.shape[0] < w:
        return 0
    else:    
        x = np.array([x for x in range(w - 1)])
        n = np.size(x)
        epochs = 100
        # L = 0.01
        error = []
        theta = np.random.randn(2,1)
        a = theta[0]
        b = theta[1]

        for i in range(epochs):
            y_pred = (a * x) + b
            error = sum((y[:-1] - y_pred) * (y[:-1] - y_pred))
            D_slope = (-2.0 / n) * sum(x * (y[:-1] - y_pred))
            D_intercept = (-2.0 / n) * sum(y[:-1] - y_pred)
            a = a - L * D_slope
            b = b - L * D_intercept

        prediction = (a * w) + b
        if (prediction > 0 and y[-1] > 0) or (prediction < 0 and y[-1] < 0):
            return 1
        else:
            return 0

# MAIN
################################################################################
print('Q2: Gradient Descent Predictions---------------------------------------')
Ls = [x/100 for x in range(1,6)]
for L in Ls:
    w = 10
    rate = df['Return'].rolling(window=(w+1), min_periods=1).apply(lambda x: custom_gradient_descent(x, (w+1), L), raw=True)
    success_rate = np.round((rate.sum()/rate.shape[0]), 4)*100
    print('Learning Rate:', L, '          Success Rate:', success_rate, '%')