#!/usr/bin/env python3
# coding: utf-8
"""
week_classifier.py
02-17-19
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

# Ensure datetime in correct format
df['Date'] = pd.to_datetime(df['Date']) - pd.to_timedelta(7, unit='d')
# Group by week, beginning sunday
weekly = df.groupby([pd.Grouper(key='Date', freq='W')])[
    'Weekday', 'Adj Close', 'Return']
# Get weekly avg std dev for comparison
std = df.groupby([pd.Grouper(key='Date', freq='W')])['Return'].std().mean()


# Helper functions
# Calculate single week's net return
def net(week):
    week = week.to_numpy()
    return (week[-1] - week[0])/week[0]


# Calculate bool for week's volatility
def vol(week):
    w_std = week.std()
    # Consider volatile if weekly standard dev > overall mean weekly std dev
    return True if w_std > std else False


# Perform group functions
def grp_vol(grp):
    grp['Vol'] = vol(grp['Return'])
    return grp


def grp_net(grp):
    grp['Net'] = net(grp['Adj Close'])
    return grp


def grp_mean_return(grp):
    grp['Avg Return'] = grp['Return'].mean()
    return grp


# Apply group calculations and return to parent df
def group_apply(data, group):
    data['Net'] = group.apply(grp_net)['Net']
    data['Vol'] = group.apply(grp_vol)['Vol']
    data['Avg Return'] = group.apply(grp_mean_return)['Avg Return']
    data['Score'] = data.apply(
        (lambda x: 1 if x['Vol'] == False and x['Net'] > 0 else 0), axis=1)


# Plot weekly summary data for specific years
def plot_weekly_summary(data, years):
    # Filter df on year
    filtered = data.loc[data['Year'].isin(years)]
    # Get weekly data
    weekly_return = filtered.groupby([pd.Grouper(key='Date', freq='W')])[
        'Return'].agg(list)
    weekly_score = filtered.groupby([pd.Grouper(key='Date', freq='W')])[
        'Score'].mean()
    # Get plot points
    x = [np.mean(np.array(i)) for i in weekly_return]
    y = [np.std(np.array(i)) for i in weekly_return]
    # Increase size for visibility
    s = [abs(ret)*2500 for ret in x]
    # Set color based on score calculated above
    c = ['green' if x > 0 else 'red' for x in weekly_score]

    # Plot
    plt.scatter(x, y, s=s, color=c)
    plt.xlabel('Mean Weekly Return')
    plt.ylabel('Weekly Standard Deviation')
    # plt.axvline(linewidth=0.25, color='b', linestyle='-')
    plt.ylim(bottom=-0.001)
    plt.show()


# MAIN
#############################################################
print('applying group functions...')
group_apply(df, weekly)
output_file = output_file.split('.')[0] + 'scored.csv'
df.to_csv(output_file, index=False)
print('plotting...')
# Plot for 2018
plot_weekly_summary(df, [2017, 2018])
