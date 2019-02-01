#!/usr/bin/env python3
# coding: utf-8
"""
monthly.py
02-01-2019
jack skrable
"""

def summarize(lines):

	# Omit header row and first day
	lines = lines[1:]
	end = len(lines)-1
	# Init dicts to store data
	data = {}
	summary = {}
	best = {'period': '', 'avg_return': 0}

	# Loop through lines in dataset
	for i, line in enumerate(lines):

		# Split string to list
		row = line.split(',')

		
		if row[2] not in data:
			data.update({row[2]: {'start': [], 'end': []}}) 

		# Log start of first month in dataset
		if i == 0:
			data[row[2]]['start'].append(float(row[9]))
		elif i == end:
			data[row[2]]['end'].append(float(row[9]))
		else:

			# Split previous row into list
			compare_row = lines[i-1].split(',')

			# If month changed
			if row[2] != compare_row[2]:
				# Log start of new month
				data[row[2]]['start'].append(float(row[9]))
				data[compare_row[2]]['end'].append(float(compare_row[9]))
							

	# Loop through holding periods
	for key in data:
		change = []
		
		for i, val in enumerate(data[key]['end']):
			try:
				change.append(((val - data[key]['start'][i])/val))
			except IndexError as e:
				print(e)

		summary.update({key: {'change': change}})


	for key in summary:

		change = summary[key]['change']

		# Get stats for changes list
		minimum = min(change)
		maximum = max(change)
		avg = (sum(change)/len(change))
		med = median(change)

		# Add to summary dict
		summary.update({key: {'summary': {'min': minimum, 'max': maximum, 'average': avg, 'median': med}}})

		if best and avg > best['avg_return']:
 			best.update({'period': key, 'avg_return': avg})
		

	print('Monthly Summary -----------------------')
	print(summary)

	print('Best  Month----------------------------')
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