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

print(data)