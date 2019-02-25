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
def net(week):
    week = week.to_numpy()
    return (week[-1] - week[0])/week[0]


def vol(week):
    w_std = week.std()
    return True if w_std > std else False


def grp_vol(grp):
    grp['Vol'] = vol(grp['Return'])
    return grp


def grp_net(grp):
    grp['Net'] = net(grp['Adj Close'])
    return grp


def grp_mean_return(grp):
    grp['Avg Return'] = grp['Return'].mean()
    return grp


def group_apply(data, group):
    # if data.loc[data['Year'].isin(years)]:
    data['Net'] = group.apply(grp_net)['Net']
    data['Vol'] = group.apply(grp_vol)['Vol']
    data['Avg Return'] = group.apply(grp_mean_return)['Avg Return']
    data['Score'] = data.apply((lambda x: 1 if x['Vol'] == False and x['Net'] > 0 else 0), axis=1)


def plot_weekly_summary(data, years):
    filtered = data.loc[data['Year'].isin(years)]
    weekly_return = filtered.groupby([pd.Grouper(key='Date', freq='W')])['Return'].agg(list)
    weekly_score = filtered.groupby([pd.Grouper(key='Date', freq='W')])['Score'].mean()
    x = [np.mean(np.array(i)) for i in weekly_return]
    y = [np.std(np.array(i)) for i in weekly_return]
    s = [abs(ret)*1000 for ret in x]
    c = ['green' if x > 0 else 'red' for x in weekly_score]

    plt.scatter(x, y, s=s, color=c)
    plt.xlabel('Mean Weekly Return')
    plt.ylabel('Weekly Standard Deviation')
    plt.axvline(linewidth=0.25, color='b-')
    plt.ylim(bottom=0)
    # plt.hlines(linewidth=0.25, color='r')
    # plt.axhline()
    plt.show()


# MAIN
#############################################################
print('applying group functions...')
group_apply(df, weekly)
print('plotting...')
plot_weekly_summary(df, [2018])


