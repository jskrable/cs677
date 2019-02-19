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

def score(week):

    net = net(week)
    vol = vol(week)
    # Some combination of the two vars returns either 1 for a good week or 0 for a bad one


df['Date'] = pd.to_datetime(df['Date']) - pd.to_timedelta(7, unit='d')

weekly = df.groupby([pd.Grouper(key='Date', freq='W')])['Weekday','Adj Close']

# Add new columns with defaults
df['Vol'] = False
df['Net'] = 0.0

# Loop thru weeks and update df columns
for key,week in weekly:

    week['Vol'] = vol(week['Adj Close'])
    week['Net'] = net(week['Adj Close'])
    df.update(week)

print(df)


