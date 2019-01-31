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

		month = row[2]
		price = float(row[9])

		try:
			data[month]['prices'].append(price)
			pass
		except KeyError as e:
			data.update({month: {'prices': [price]}})



	for key in data:

		prices = data[key]['prices']
		changes = []
		for i, val in enumerate(prices):
			change = ((val - prices[i-1])/prices[i-1])
			changes.append(change)

		data[key].update({'changes': changes})
		
		# print('DATA--------------')
		# print(data)

		minimum = min(data[key]['changes'])
		maximum = max(data[key]['changes'])
		avg = (sum(data[key]['changes'])/len(data[key]['changes']))
		med = median(data[key]['changes'])

		summary.update({key: {'summary': {'min': minimum, 'max': maximum, 'average': avg, 'median': med}}})

	print('Monthly Data -----------------------------')
	print(summary)
		


def median(data):

	data.sort()
	mid = int((len(data)-1)/2)

	if len(data) % 2:
		return data[mid]
	else:
		return (data[mid-1] + data[mid])/2
