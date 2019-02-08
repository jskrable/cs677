#!/usr/bin/env python3
# coding: utf-8
"""
short_ma.py
02-01-2019
jack skrable
"""

def summarize(lines):

	# Omit header row
	lines = lines[1:]

	# Init trades list
	buy = -1
	trades = []

	# Loop through lines in dataset
	for i, line in enumerate(lines):
		
		# Split string to list
		today = line.split(',')

		close = float(today[9])
		sma = float(today[11])

		if close > sma:
			if buy == -1:
				buy = close
		elif buy != -1:
			trades.append((close-buy)/close)
			buy = -1
			

	# print('TRADES -------------------')
	# print(trades)
	# return trades

	total = len(trades)

	pos = [x for x in trades if x > 0]
	neg = [x for x in trades if x < 0]
	pos_cnt = len(pos)
	neg_cnt = len(neg)
	pos_avg = sum(pos)/pos_cnt
	neg_avg = sum(neg)/neg_cnt

	summary = {'trades': total, 'profitable': pos_cnt, 'avg profit': pos_avg, 'lossses': neg_cnt, 'avg loss': neg_avg}


	return summary