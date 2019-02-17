#!/usr/bin/env python3
# coding: utf-8
"""
bollinger.py
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

# Get lists of w and k values
W = [x for x in range(10,101,10)]
K = [0.1*x for x in range(0,31,5)]

# Init vars 
shares = 0
results = {}


# Loop through Ws
for w in W:
	# Update df to inlude W-moving avg and std
	df['W_MA'] = df['Adj Close'].rolling(window=w, min_periods=1).mean()
	df['W_Std'] = df['Adj Close'].rolling(window=w, min_periods=1).std()

	# Loop through dataset
	for i, row in df.iterrows():

		close = row['Adj Close']
		ma = row['W_MA']
		std = row['W_Std']

		# print(pd.isnull(pd.Series(close,ma,std)))

		for k in K:

			print(close)
			print(ma)
			print(std)

			if close < (ma - (k*std)) and shares == 0:
				shares = 100/df['Adj Close']


			elif close > (ma - (k*std)) and shares > 0:
				net = shares * 100
				shares = 0

			else:
				net = 0

			key = str(w)+'_'+str(k)
			results.update({key: net})

# print(results)









# df['W_MA'] = df['Adj Close'].rolling(window=W, min_periods=1).mean()
# df['W_Std'] = df['Adj Close'].rolling(window=W, min_periods=1).std()

