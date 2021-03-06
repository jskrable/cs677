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
import matplotlib.pyplot as plt

ticker = 'SYK'

# Get path of dataset
if os.name == 'posix':
    input_dir = r'/home/jskrable/code/cs677/datasets'
else:
    input_dir = r'C:\Users\jskrable\code\cs677\datasets'
output_file = os.path.join(input_dir, ticker + '.csv')

# Read file into pandas dataframe
df = pd.read_csv(output_file)


def bollinger(df):

	# Get lists of w and k values
	W = [x for x in range(10,101,10)]
	K = [0.1*x for x in range(0,31,5)]

	# Init vars 
	shares = 0
	results = {}

	# Making trades
	print('Simulating trades...')
	for w in W:
		# Update df to inlude W-moving avg and std
		df['W_MA'] = df['Adj Close'].rolling(window=w, min_periods=1).mean()
		df['W_Std'] = df['Adj Close'].rolling(window=w, min_periods=1).std()

		for k in K:
			# Print current simulation to console
			print('Trying w=',w,'k=',k)
			key = str(w)+'_'+str(k)
			# Create new entry in results
			results.update({key: {'values': []}})

			# Loop through dataset
			for i, row in df.iterrows():

				close = row['Adj Close']
				ma = row['W_MA']
				std = row['W_Std']

				try: 
					# print(close < ma - k*std and shares == 0)
					if close < (ma - (k*std)) and shares == 0:
						shares = 100/close
						# print('Bought',shares,'shares')

					elif close > (ma + (k*std)) and shares > 0:
						net = 100-(shares*close)
						results[key]['values'].append(net)
						shares = 0
					
				except ValueError as e:
					print(e)

	# Get average return from results
	print('Analyzing results...')
	for key in results:
		returns = np.asarray(results[key]['values'])
		avg = np.average(returns) if len(returns) > 0 else 0.0
		results[key].update({'avg':avg})

	# Return trade results
	return results


def scatterplot(data):

	# Plot points
	x = [x.split('_')[0] for x, y in data.items()]
	y = [x.split('_')[1] for x, y in data.items()]
	# Set color of points
	c = ['green' if y['avg'] > 0 else 'red' for x,y in data.items()]
	# Increase size for visibility
	s = [abs(y['avg']*10) for x, y in data.items()]

	print('Displaying scatterplot...')
	plt.scatter(x,y,s=s,color=c)
	plt.xlabel('W')
	plt.ylabel('k')
	plt.show()




scatterplot(bollinger(df))
