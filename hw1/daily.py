#!/usr/bin/env python3
# coding: utf-8
"""
daily.py
02-01-2019
jack skrable
"""

def summarize(lines):

	# Omit header row and first day
	lines = lines[2:]
	# Init dicts to store data
	summary = {}
	best = {'period': '', 'avg_return': 0}

	# Loop through lines in dataset
	for i, line in enumerate(lines):

		# Split string to list
		row = line.split(',')
		# Split previous row into list
		compare_row = lines[i-1].split(',')

		day = row[4]

		change = ((float(row[9])- float(compare_row[9])) / float(row[9]))


		# Populate change array for each day of the week
		try:
			summary[day]['change'].append(change)
			pass
		except KeyError as e:
			summary.update({day: {'change': [change]}})


	# Loop through holding periods
	for key in summary:

		change = summary[key]['change']

		# Get stats for changes list
		minimum = min(summary[key]['change'])
		maximum = max(summary[key]['change'])
		avg = (sum(summary[key]['change'])/len(summary[key]['change']))
		med = median(summary[key]['change'])

		# Add to summary dict
		summary.update({key: {'summary': {'min': minimum, 'max': maximum, 'average': avg, 'median': med}}})

		if best and avg > best['avg_return']:
			best.update({'period': key, 'avg_return': avg})
		

	print('Daily Summary -----------------------')
	print(summary)

	print('Best  Day----------------------------')
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