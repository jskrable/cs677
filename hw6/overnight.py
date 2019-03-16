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


# Helper function to calculate overnight change for a row
def overnight_change(row):
    return row.Open - df.iloc[row.name-1].Close


# Helper function to calc overnight return for a row
def overnight_return(row):
    prev_close = df.iloc[row.name-1].Close
    return (row.Open - prev_close) / prev_close


# Trading strategy based on overnight price change inertia
def on_inertia(row, R):
    global shares
    if shares == 0 and row['Overnight_Change'] > R:
        shares = 100 / row.Open

    elif shares > 0 and row['Overnight_Change'] <= R:
        net = (row.Open -row.Close) / shares
        shares = 0
        # return_df.append({'date': row.Date, 'return': net}, ignore_index=True)
        return net


# Runs overnight inertia strategy without any minimum change parameter
def naive_on_inertia(df):
    returns = df.apply(lambda x: on_inertia(x, 0), axis=1)
    returns = returns[returns.isnull() == False]
    return returns


# Plots return columns from dataframe
def plot_returns(df):
    x = df['Overnight_Return'].values
    y = df['Return'].values
    corr = np.round(df[['Overnight_Return','Return']].corr().Return.Overnight_Return, 5)
    plt.scatter(x,y)
    plt.xlabel('Overnight Returns')
    plt.ylabel('Daily Returns')
    plt.title('Daily Returns vs. Overnight Returns')
    plt.text(-0.04, 0.07, 'corr = ' + str(corr))
    plt.show()


# Plots results of strategy from dict
def plot_results(results):
    x = [x for x in results.keys()]
    y = [y for y in results.values()]
    plt.plot(x, y, 'o:')
    plt.ylabel('Mean Return')
    plt.xlabel('R-value')
    plt.title('Mean Return vs. R-value')
    plt.show()


# Runs overnight inertia strategy with a range of min change parameters
def minimum_rs(df):
    Rs = [r for r in range(-10,11)]
    results = {}
    for R in Rs:
        print('Simulating with r =',R)
        returns = df.apply(lambda x: on_inertia(x, R), axis=1)
        returns = returns[returns.isnull() == False]
        results.update({R: returns.mean()})

    return results


# MAIN
################################################################
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

# Add overnight return column
print('Processing overnight columns...')
df['Overnight_Return'] = df.apply(overnight_return, axis=1)
df['Overnight_Change'] = df.apply(overnight_change, axis=1)

# Drop first row with bad value for overnight
df = df[1:]

# Globals
shares = 0
results = {}

# Q1
print('Q1: Naive Overnight Inertia --------------------------------------------')
naive = naive_on_inertia(df)
# plot_returns(naive)

# Q2
print('Q2: Algorithm Analysis -------------------------------------------------')
print('Displaying plot...')
plot_returns(df)

# Q3
print('Q3: Minimum Return Overnight Inertia -----------------------------------')
R_results = minimum_rs(df)
print('Displaying plot...')
plot_results(R_results)
