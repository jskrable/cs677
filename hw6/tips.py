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

