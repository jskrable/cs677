#!/usr/bin/env python3
# coding: utf-8
"""
tips.py
02-25-19
jack skrable
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Get path of set
if os.name == 'posix':
    input_dir = r'/home/jskrable/code/cs677/datasets'
else:
    input_dir = r'C:\Users\jskrable\code\cs677\datasets'
output_file = os.path.join(input_dir, 'tips.csv')

# Read file into pandas dataframe
print('Reading',output_file,'...')
df = pd.read_csv(output_file)

df['tip_percent'] = (df['tip'] / df['total_bill'])*100

# q1
df.groupby('time')['tip_percent'].mean()

#q2
df.groupby(['day','time'])['tip_percent'].mean()

#q5
df['smoker'].describe()['top']
np.round(((len(df) - df['smoker'].describe()['freq']) / len(df))*100,2)
