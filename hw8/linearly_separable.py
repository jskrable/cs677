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
# weekly = weekly.reset_index()

x = [x for x in weekly.index]
y = [y for y in weekly.Mean]
c = ['green' if x > 0 else 'red' for x in weekly.Score]
plt.scatter(x, y, color=c)
# plt.xlabel('Time')
plt.ylabel('Mean Weekly Return')
plt.show()
