#!/usr/bin/env python3
# coding: utf-8
"""
short_long.py
02-18-19
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


def short_long(df):

	# Get lists of w and k values
	WS = np.arange(1,26)
	WL = np.arange(25,51)

	# Init vars 
	shares = 0
	results = {}

	# Making trades
	print('Simulating trades...')
	for i, ws in np.ndenumerate(WS):

		for i, wl in np.ndenumerate(WL):
			# Update df to inlude W-short and long moving avg
			df['W_short'] = df['Adj Close'].rolling(window=ws, min_periods=1).mean()
			df['W_long'] = df['Adj Close'].rolling(window=wl, min_periods=1).mean()

			# Print current simulation to console
			print('Trying WS=',ws,'WL=',wl)
			key = str(ws)+'_'+str(wl)
			# Create new entry in results
			results.update({key: {'values': []}})

			# Loop through dataset
			for i, row in df.iterrows():

				close = row['Adj Close']
				ma_S = row['W_short']
				ma_L = row['W_long']

				try: 
					# print(close < ma - k*std and shares == 0)
					if ma_S > ma_L and shares == 0:
						shares = 100/close
						# print('Bought',shares,'shares')

					elif ma_S < ma_L and shares > 0:
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

# Plot the data
def scatterplot(data):

	# Plot points
	x = [x.split('_')[0] for x, y in data.items()]
	y = [x.split('_')[1] for x, y in data.items()]
	# Set color of points
	c = ['green' if y['avg'] > 0 else 'red' for x,y in data.items()]
	# Increase size for visibility
	s = [abs(y['avg']*20) for x, y in data.items()]

	print('Displaying scatterplot...')
	plt.scatter(x,y,s=s,color=c)
	plt.xlabel('Short MA Window')
	plt.ylabel('Long MA Window')
	plt.show()




scatterplot(short_long(df))
