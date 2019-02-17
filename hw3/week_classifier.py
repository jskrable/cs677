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
data = pd.read_csv(output_file)




