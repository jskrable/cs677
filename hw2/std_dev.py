#!/usr/bin/env python3
# coding: utf-8
"""
std_dev.py
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
output_file = os.path.join(input_dir, ticker + '.csv')

data = pd.read_csv(output_file)
close = data['Adj Close']

mean = close.mean()
# print('Mean: ', mean)
size = len(close)
num = map(lambda x : (x - mean)**2, close)
std_cust = (sum(list(map(lambda x : (x - mean)**2, close)))/size)**(0.5)
std = close.std()
# print('Custom Standard Dev: ', std_cust)
# print('Standard Dev: ', std)

outliers = [x for x in close if abs(mean - x) > 2*std]

print(size)
print(len(outliers))
print((len(outliers)/size)*100, '% outside')
