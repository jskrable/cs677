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


# Helper function to calc overnight return for a row
def overnight_return(row):
    prev_close = df.iloc[row.name-1].Close
    return (row.Open - prev_close) / prev_close


# Trading strategy based on overnight price change inertia
def day_trading(row, R):

    if R >= 0:
        if row['Overnight_Return']*100 > R:
            trade = ((100 / row.Open) * row.Close) - 100
            return trade

    elif R < 0:
        if row['Overnight_Return']*100 < R:
            trade = ((100 / row.Close) * row.Open) - 100
            return trade


# Plots results of strategy from dict
def plot_results(results):
    x = [x for x in results.keys()]
    # Total P/L
    y1 = [y.sum() for y in results.values()]
    # Fraction of profitable trades
    y2 = [((y > 0).sum() / len(y))*100 for y in results.values()]
    # First plot
    plt.plot(x, y1, 'o:')
    plt.ylabel('Profit/Loss')
    plt.xlabel('R-value')
    plt.title('Total P/L vs. R-value')
    plt.show()
    # Second plot
    plt.plot(x, y2, 'o:')
    plt.ylabel('% of Profitable Trades')
    plt.xlabel('R-value')
    plt.title('Profitable Trades vs. R-value')
    plt.show()


# Runs overnight inertia strategy with a range of min change parameters
def simulate_trades(df):
    Rs = [r for r in range(-5,6)]
    results = {}
    # For each possible R-value
    for R in Rs:
        print('Simulating with r =',R)
        # Apply day traading strategy
        returns = df.apply(lambda x: day_trading(x, R), axis=1)
        returns = returns[returns.isnull() == False]
        results.update({R: returns.values})

    return results


# MAIN
################################################################
ticker = 'SYK'

# Get path of dataset
if os.name == 'posix':
    input_dir = r'/home/jskrable/code/cs677/datasets'
else:
    input_dir = r'C:\Users\jskrable\code\cs677\datasets'
output_file = os.path.join(input_dir, ticker + '.csv')

# Read file into pandas dataframe
print('Reading data file...')
df = pd.read_csv(output_file)

# Add overnight return column
print('Calculating overnight returns...')
df['Overnight_Return'] = df.apply(overnight_return, axis=1)

# Q1
print('Day trading based on overnight inertia...')
results = simulate_trades(df)
print('Plotting results...')
plot_results(results)
print('Complete')
