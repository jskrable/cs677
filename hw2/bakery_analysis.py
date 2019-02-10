#!/usr/bin/env python3
# coding: utf-8
"""
bakery_analysis.py
02-10-2019
jack skrable
"""

import os
import pandas as pd
 
# Get path of dataset
if os.name == 'posix':
    input_dir = r'/home/jskrable/code/cs677/datasets'
else:
    input_dir = r'C:\Users\jskrable\code\cs677\datasets'
output_file = os.path.join(input_dir, 'BreadBasket_DMS_output.csv')

# Read file into pandas dataframe
data = pd.read_csv(output_file)

period = data.groupby('Period')
hour = data.groupby('Hour')
day = data.groupby('Weekday')

print(period['Item_Price'].sum())
print(hour['Item_Price'].sum())
print(day['Item_Price'].sum())