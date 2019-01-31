#!/usr/bin/env python3
# coding: utf-8
"""
summarize.py
01-31-2019
jack skrable
"""


# Function to summarize % change is adj. close if held for a specific interval
def interval(lines, interval):

	# Omit header row
	lines = lines[1:]
	# Init dicts to store data
	data = {}
	summary = {}
	best = {'period': '', 'avg_return': 0}

	# Loop through lines in dataset
	for line in lines:
		# Split csv string
		row = line.split(',')
		# Choose time span to hold stock for
		hold = row[4] if interval == 'day' else row[2]
		# Get adj. closing price
		price = float(row[9])

		# Populate prices array for each holding period
		try:
			data[hold]['prices'].append(price)
			pass
		except KeyError as e:
			data.update({hold: {'prices': [price]}})


	# Loop through holding periods
	for key in data:

		prices = data[key]['prices']
		changes = []
		# Loop through prices for period
		for i, val in enumerate(prices):
			# Ignore first price
			change = 0 if i == 0 else ((val - prices[i-1])/prices[i-1])
			changes.append(change)

		# Add changes to holding period
		data[key].update({'changes': changes})
		
		# print('DATA--------------')
		# print(data)

		# Get stats for changes list
		minimum = min(data[key]['changes'])
		maximum = max(data[key]['changes'])
		avg = (sum(data[key]['changes'])/len(data[key]['changes']))
		med = median(data[key]['changes'])

		# Add to summary dict
		summary.update({key: {'summary': {'min': minimum, 'max': maximum, 'average': avg, 'median': med}}})

		if best and avg > best['avg_return']:
			best.update({'period': key, 'avg_return': avg})
		

	# print('Data -----------------------')
	# print(summary)

	print('Best -----------------------')
	print(best)

	return summary
		

# Function to get median value of a list
def median(data):
	data.sort()
	mid = int((len(data)-1)/2)
	# If even list return avg of two middle values
	if len(data) % 2:
		return data[mid]
	else:
		return (data[mid-1] + data[mid])/2