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

		print('Trying w=',w,'k=',k)
		key = str(w)+'_'+str(k)
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

# Analyzing results
print('Analyzing results...')
for key in results:
	returns = np.asarray(results[key]['values'])
	avg = np.average(returns)
	results[key].update({'avg':avg})


# Plot points
x = [x.split('_')[0] for x, y in results.items()]
y = [x.split('_')[1] for x, y in results.items()]
s = [y['avg'] for x, y in results.items()]
c = ['green' if val > 0 else 'red' for i, val in enumerate(s)]
print(c)

print('Displaying scatterplot...')
# for i, val in enumerate(s):
# 	print(x[i],y[i],val)
# 	color = 'green' if val > 0 else 'red'
# 	print(val > 0)
# 	print(color)
# 	plt.scatter(x[i], y[i], s=(s[i]*2), c=('green' if val > 0 else 'red'))

plt.scatter(x,y,s=s,c=c)
plt.show()












# df['W_MA'] = df['Adj Close'].rolling(window=W, min_periods=1).mean()
# df['W_Std'] = df['Adj Close'].rolling(window=W, min_periods=1).std()

