#!/usr/bin/env python3
# coding: utf-8
"""
last_digit.py
02-08-2019
jack skrable
"""

import os
import pandas as pd
 
ticker = 'SYK'

# Get path of dataset
if os.name == 'posix':
    input_dir = r'/home/jskrable/code/cs677/datasets'
else:
    input_dir = r'C:\Users\jskrable\code\cs677\datasets'
output_file = os.path.join(input_dir, ticker + '_digit_analysis.csv')

data = pd.read_csv(output_file)
data['error'] = data.apply(lambda x : abs(x['digit_frequency'] - 0.1), axis=1)

# print(data)

max_abs = max(data['error'])
med_abs = data['error'].median()
mean_abs = data['error'].mean()
root_mean_sq = (sum(data['error']**2)/len(data))**(0.5)

print('Max Absolute Error: ', max_abs)
print('Median Absolute Error: ', med_abs)
print('Mean Absolute Error: ', mean_abs)
print('Root Mean Squared Error: ', root_mean_sq)

