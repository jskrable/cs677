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


def busiest(data, group):

	interval = data.groupby(group).agg({'Transaction':'count'})
	interval_maxkey = interval['Transaction'].idxmax()
	interval_max = interval['Transaction'].max()

	print('Busiest interval for',group,'was',interval_maxkey,'with',interval_max,'transactions.')

def profitable(data, group):

	interval = data.groupby(group).agg({'Item_Price':'sum'})
	interval_maxkey = interval['Item_Price'].idxmax()
	interval_max = interval['Item_Price'].max().round(2)
	print('Most profitable interval for',group,'was',interval_maxkey,'with',interval_max,'in revenue.')

print('Busiest-------------------------------------------------------------')
busiest(data, 'Period')
busiest(data, 'Hour')
busiest(data, 'Weekday')
print('Most Profitable-----------------------------------------------------')
profitable(data, 'Period')
profitable(data, 'Hour')
profitable(data, 'Weekday')

hour = data.groupby('Hour')
day = data.groupby('Weekday')

# print('Busiest -----------------------')
# print(max(hour['Transaction'].count()).key())
# print(hour['Transaction'].count())
# print(day['Transaction'].count())
# print(period['Transaction'].count())
# print('Most Profitable----------------')
# print(hour['Item_Price'].count())
# print(day['Item_Price'].count())
# print(period['Item_Price'].count())