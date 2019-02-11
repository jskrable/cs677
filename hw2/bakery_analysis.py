#!/usr/bin/env python3
# coding: utf-8
"""
bakery_analysis.py
02-10-2019
jack skrable
"""

import os
import math
import pandas as pd
 
# Get path of dataset
if os.name == 'posix':
    input_dir = r'/home/jskrable/code/cs677/datasets'
else:
    input_dir = r'C:\Users\jskrable\code\cs677\datasets'
output_file = os.path.join(input_dir, 'BreadBasket_DMS_output.csv')

# Read file into pandas dataframe
data = pd.read_csv(output_file)

# print(data.groupby('Item').agg({'Item': 'count'}))
# Concat date field
# TODO find more efficient way to do this
data['Date'] = data.apply(lambda x : str(x['Year'])+str(x['Month'])+str(x['Day']), axis=1)


# Function to get avg group size based on number of drinks per transaction
def group(data):
	# Get number of drinks per transaction
	group = data.groupby('Transaction')['Category'].apply(lambda x : (x == 'Drink').sum()).reset_index(name='Drinks')
	print('Average group size per transaction based on drink count',round(group['Drinks'].mean(),4))


# Function to analyze based on type of product
def category(data):
	mean_prices = data.groupby('Category').agg({'Item_Price': 'mean'})
	for i, x in mean_prices['Item_Price'].items():
		print('Average price for',i,'is',round(x,2))
	rev = data.groupby('Category').agg({'Item_Price': 'sum'})
	for i, x in rev['Item_Price'].items():
		print('Total revenue for',i,'is',round(x,2))

	print('Shop makes most money from',rev['Item_Price'].idxmax())


# Function to get the minimum number of baristas per day
def baristas(data):
	# Transaction total by weekday
	transactions = data.groupby('Weekday').Transaction.nunique()
	# Number of each weekday in dataset
	days = data.groupby('Weekday').Date.nunique()
	# Join to dataframe
	weekdays = transactions.to_frame().join(days.to_frame())
	# Loop through frame
	for i, x in weekdays['Transaction'].items():
		# Get barista count assuming each can take 60 trxns per day
		baristas = math.ceil((x/weekdays['Date'][i])/60)
		day = i + 's'
		print('On',day,baristas,'baristas are needed.')


# Function to get popularity of menu items
def popularity(data):
	items = data.groupby('Item').agg({'Item': 'count'})
	print('Most popular item is',items['Item'].idxmax(),'with',items['Item'].max(),'purchases')
	print('Least popular item is',items['Item'].idxmin(),'with',items['Item'].min(),'purchases')

	# TODO try to do this with pandas?
	combos = {}
	# Populate dict with key of transaction id and list of items bought
	for i,x in data.iterrows():
		try: 
			combos[x['Transaction']].append(x['Item'])
		except KeyError:
			combos.update({x['Transaction']: [x['Item']]})

    # Drop anything with only one item
	combos = {x:combos[x] for x in combos.keys() if len(combos[x]) > 1}
	# Sort list values
	combos = {x:sorted(combos[x]) for x in combos.keys()}

	# New dict with counts based on stringified item lists
	combo_count = {}
	for x in combos:
		try:
			combo_count[str(combos[x])] += 1
		except KeyError:
			combo_count.update({str(combos[x]): 1})

    # Get most popular combination
	pop_combo = max(combo_count, key=combo_count.get)
	pop_combo_count = combo_count[pop_combo]
	print('Most popular combination of items is',pop_combo,'with',pop_combo_count,'purchases')


# Function to get business by time interval
def busiest(data, group):
	interval = data.groupby(group).agg({'Transaction':'count'})
	interval_maxkey = interval['Transaction'].idxmax()
	interval_max = interval['Transaction'].max()
	print('Busiest interval for',group,'was',interval_maxkey,'with',interval_max,'transactions.')


# Function to get profitability by time interval
def profitable(data, group):
	interval = data.groupby(group).agg({'Item_Price':'sum'})
	interval_maxkey = interval['Item_Price'].idxmax()
	interval_max = interval['Item_Price'].max().round(2)
	print('Most profitable interval for',group,'was',interval_maxkey,'with',interval_max,'in revenue.')


# Function to print output to console
def output(data):

	print('Busiest-------------------------------------------------------------')
	busiest(data, 'Period')
	busiest(data, 'Hour')
	busiest(data, 'Weekday')
	print('Most Profitable-----------------------------------------------------')
	profitable(data, 'Period')
	profitable(data, 'Hour')
	profitable(data, 'Weekday')
	print('Item Popularity-----------------------------------------------------')
	popularity(data)
	print('Hiring--------------------------------------------------------------')
	baristas(data)
	print('Categories----------------------------------------------------------')
	category(data)
	print('Group Size----------------------------------------------------------')
	group(data)

output(data)

