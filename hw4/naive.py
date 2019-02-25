#!/usr/bin/env python3
# coding: utf-8
"""
naive.py
02-24-19
jack skrable
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cProfile

ticker = 'SYK'

# Get path of dataset
if os.name == 'posix':
    input_dir = r'/home/jskrable/code/cs677/datasets'
else:
    input_dir = r'C:\Users\jskrable\code\cs677\datasets'
output_file = os.path.join(input_dir, ticker + '.csv')

# Read file into pandas dataframe
df = pd.read_csv(output_file)

# init flags
shares = 0
results = {}

# loop thru df rows
for i, row in df.iterrows():

    if shares == 0 and row['Return'] > 0:
        shares = 100/(row['Adj Close'])

    if shares > 0 and row['Return'] < 0:
        net = 100 - (shares * (row['Adj Close']))
        results.update({i: net})
        shares = 0


# Plot points
x = [x for x in results.keys()]
y = [y for x,y in results.items()]

# Show plot
plt.plot(x,y)
plt.xlabel('Time')
plt.ylabel('Return')
plt.title('Naive Strategy')
plt.show()


