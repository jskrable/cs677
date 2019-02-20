#!/usr/bin/env python3
# coding: utf-8
"""
week_classifier.py
02-17-19
jack skrable
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cProfile

ticker = 'SYK'

# Get path of dataset
if os.name == 'posix':
    input_dir = r'/home/jskrable/code/cs677/datasets'
else:
    input_dir = r'C:\Users\jskrable\code\cs677\datasets'
output_file = os.path.join(input_dir, ticker + '.csv')

# Read file into pandas dataframe
df = pd.read_csv(output_file)

# Get stats for whole dataset
std = df['Adj Close'].std()

# Helper functions
def net(week):
    week = week.to_numpy()
    return (week[-1] - week[0])

def vol(week):
    # Change comparison to std dev of dataset up to this point?
    w_std = week.std()
    return True if w_std > std else False

def grp_vol(grp):
    grp['Vol'] = vol(grp['Adj Close'])
    return grp

def grp_net(grp):
    grp['Net'] = net(grp['Adj Close'])
    return grp

def score(week):
    net = net(week)
    vol = vol(week)
    # Some combination of the two vars returns either 1 for a good week or 0 for a bad one

print('Grouping by week-------------------------------------')
# Ensure datetime in correct format
df['Date'] = pd.to_datetime(df['Date']) - pd.to_timedelta(7, unit='d')
# Group by week, beginning sunday
weekly = df.groupby([pd.Grouper(key='Date', freq='W')])['Weekday','Adj Close','Return']

print('Applying group functions-----------------------------')
def group_apply(data):
    df['Net'] = weekly.apply(grp_net)['Net']
    df['Vol'] = weekly.apply(grp_vol)['Vol']

weekly.apply(grp_net)['Net'].hist()
plt.show()
group_apply(weekly)



