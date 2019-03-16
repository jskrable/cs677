#!/usr/bin/env python3
# coding: utf-8
"""
overnight.py
03-15-19
jack skrable
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ticker = 'SYK'

# Get path of set
if os.name == 'posix':
    input_dir = r'/home/jskrable/code/cs677/datasets'
else:
    input_dir = r'C:\Users\jskrable\code\cs677\datasets'
output_file = os.path.join(input_dir, ticker + '.csv')

# Read file into pandas dataframe
print('Reading data file...')
df = pd.read_csv(output_file)


# Helper function to calculate overnight return for a row
def overnight(row):
    prev_close = df.iloc[row.name-1].Close
    return (row.Open - prev_close) / prev_close

# Add overnight return column
df['Overnight'] = df.apply(overnight, axis=1)

# Drop first row with bad value for overnight
df = df[1:]

shares = 0
results = {}


def on_inertia(row):
    global shares
    if shares == 0 and row.Overnight > 0:
        shares = 100 / row.Open

    elif shares > 0 and row.Overnight <= 0:
        net = (row.Open -row.Close) / shares
        shares = 0
        results.update({row.name : net})

df.apply(on_inertia)


def plot_returns(df):

    x = df.Overnight.values
    y = df.Return.values
    plt.scatter(x,y)
    plt.show()


def plot_results(results):

    x = [x for x in results.keys()]
    y = [y for y in results.values()]
    plt.plot(x,y)
    plt.show()




