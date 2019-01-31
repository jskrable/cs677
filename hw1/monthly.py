# coding: utf-8
"""
daily.py
01-30-2019
jack skrable
"""

def strategy(lines):

	lines = lines[1:]
	data = {}
	summary = {}

	for line in lines:
		row = line.split(',')

		day = row[2]
		price = float(row[9])

		try:
			data[day].append(price)
			pass
		except KeyError as e:
			data.update({day: [price]})

	for key in data:

		prices = data[key]
		for i, val in enumerate(prices):
			print('Current: ',val)
			print('Prev: ', prices[i-1])
			change = ((val - prices[i-1])/prices[i-1])
			print('Change: ',change)


		minimum = min(data[key])
		maximum = max(data[key])
		avg = (sum(data[key])/len(data[key]))
		med = median(data[key])

		summary.update({key: {'min': minimum, 'max': maximum, 'average': avg, 'median': med}})

	#print(summary)
		


def median(data):

	data.sort()
	mid = int((len(data)-1)/2)

	if len(data) % 2:
		return data[mid]
	else:
		return (data[mid-1] + data[mid])/2
