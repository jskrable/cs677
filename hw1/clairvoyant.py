#!/usr/bin/env python3
# coding: utf-8
"""
clairvoyant.py
02-04-2019
jack skrable
"""

def predict(lines):

	# Omit header row
	lines = lines[1:]

	# Init trades list
	buy = -1
	trades = []
	shares = 0

	low = [100000, 0]
	high = [0, 0]

	# Loop through lines in dataset
	for i, line in enumerate(lines):
		
		# Split string to list
		today = line.split(',')
		close = float(today[9])

		# get low and high closes
		if close < low[0]:
			low[0] = close
			low[1] = i 
		if close > high[0]:
			high[0] = close
			high[1] = i 

	for i, line in enumerate(lines):

		# Split string to list
		today = line.split(',')
		close = float(today[9])

		# Buy on lowest close 
		if i == low[1]:
			shares = 100 * close
			buy_day = today[0]

		# Sell on highest close
		if i == high[1]:
			shares = (100 * close) - shares
			sell_day = today[0]

	summary = {'buy_day': buy_day, 'sell_day': sell_day, 'profit': shares}

	return summary