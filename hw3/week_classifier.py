#!/usr/bin/env python3
# coding: utf-8
"""
strategies.py
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
    w_std = week.std()
    return True if w_std > std else False


df['Date'] = pd.to_datetime(df['Date']) - pd.to_timedelta(7, unit='d')

weekly = df.groupby([pd.Grouper(key='Date', freq='W')])['Weekday','Adj Close']

for key,item in weekly:

    item['Vol'] = vol(item['Adj Close'])
    item['Net'] = net(item['Adj Close'])
    print(item)
    # df.update(item)

print(df)

    
    

# df = df.groupby(['Name', pd.Grouper(key='Date', freq='W-MON')])['Quantity']
#        .sum()
#        .reset_index()
#        .sort_values('Date')

