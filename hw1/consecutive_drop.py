#!/usr/bin/env python3
# coding: utf-8
"""
consecutive_drop.py
02-01-2019
jack skrable
"""

def summarize(lines, w):

	# Omit header row
	lines = lines[1:]

	# Init counters
	loss_cnt = 0
	buy = -1
	trades = []

	# Loop through lines in dataset
	for i, line in enumerate(lines):
		
		# Split string to list
		today = line.split(',')
		yest = lines[i-1].split(',')


		close = float(today[9])
		last_close = float(yest[9])

		delta = last_close - close

		if buy == (i-1):
			trades.append((close-last_close)/close)

		if delta < 0:
			loss_cnt += 1

			if loss_cnt == w:

				buy = i
				loss_cnt = 0

	# print('TRADES -------------------')
	# print(trades)
	#return trades

	total = len(trades)

	pos = [x for x in trades if x > 0]
	neg = [x for x in trades if x < 0]
	pos_cnt = len(pos)
	neg_cnt = len(neg)
	pos_avg = sum(pos)/pos_cnt
	neg_avg = sum(neg)/neg_cnt

	summary = {w: {'trades': total, 'profitable': pos_cnt, 'avg profit': pos_avg, 'lossses': neg_cnt, 'avg loss': neg_avg}}

	return summary









		