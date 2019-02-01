#!/usr/bin/env python3
# coding: utf-8
"""
strategies.py
01-31-2019
jack skrable
"""

import os
import daily
import monthly
 
ticker = 'SYK'

# Get path of dataset
if os.name == 'posix':
    input_dir = r'/home/jskrable/code/cs677/datasets'
else:
    input_dir = r'C:\Users\jskrable\code\cs677\datasets'
output_file = os.path.join(input_dir, ticker + '.csv')

# Read data file
with open(output_file) as f:
    lines = f.read().splitlines()

    # Run summarize for day and month
    daily = daily.summarize(lines)
    monthly = monthly.summarize(lines)