#!/usr/bin/env python3
# coding: utf-8
"""
portfolio.py
04-29-19
jack skrable
"""

import os
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

tickers = ['SYK', 'BSX']

# Get path of set
if os.name == 'posix':
    input_dir = r'/home/jskrable/code/cs677/datasets'
else:
    input_dir = r'C:\Users\jskrable\code\cs677\datasets'
output_files = [os.path.join(input_dir, ticker + '.csv') for ticker in tickers]

# Read labelled file into pandas dataframe
print('Reading data files...')
df_syk = pd.read_csv(output_files[0])
df_bsx = pd.read_csv(output_files[1])

df = pd.DataFrame.from_records(
    {'date': df_syk.Date, 'syk': df_syk.Return, 'bsx': df_bsx.Return})

np.cov(df.syk, df.bsx)


def weighted_returns(Wb, Ws):
    print('BSX Weight: {} SYK Weight: {}'.format(Wb, Ws))
    ret_b = df.bsx.apply(lambda x: x*Wb).sum()
    ret_s = df.syk.apply(lambda x: x*Ws).sum()
    return ret_b + ret_s


def markowitz_simulation(df):
    weights = [np.round(x * .10, 1) for x in range(11)]
    print('Simulating portfolios...')
    return {np.round(w, 1): weighted_returns(w, np.round(1-w, 1)) for w in weights}


def plot_results(results):
    sns.lineplot(list(results.keys()), list(results.values()))
    plt.xlabel('BSX share of portfolio')
    plt.ylabel('% return of portfolio')
    plt.show()

def covariance(df):
    covar = np.cov(df.bsx, df.syk)
    print('BXS to SYK Co-variance: {}'.format(covar[0][0]))
    print('SYK to BXS Co-variance: {}'.format(covar[1][1]))

    
# MAIN
##############################################################################
covariance(df)
plot_results(markowitz_simulation(df))