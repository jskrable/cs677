#!/usr/bin/env python3
# coding: utf-8
"""
tips.py
03-14-19
jack skrable
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Get path of set
if os.name == 'posix':
    input_dir = r'/home/jskrable/code/cs677/datasets'
else:
    input_dir = r'C:\Users\jskrable\code\cs677\datasets'
output_file = os.path.join(input_dir, 'tips.csv')

# Hide setting with copy warning
pd.options.mode.chained_assignment = None  # default='warn'

# Read file into pandas dataframe
print('Reading',output_file,'...')
df = pd.read_csv(output_file)

df['tip_percent'] = (df['tip'] / df['total_bill'])*100

# q1
print('Q1 ---------------------------------------------------------')
meal_avg = df.groupby('time')['tip_percent'].mean()
print('On average, tips are higher at',meal_avg.idxmax(),'\n')
print(meal_avg,'\n')

#q2
print('Q2 ---------------------------------------------------------')
day_meal_avg = df.groupby(['day','time'])['tip_percent'].mean()
day, meal = day_meal_avg.idxmax()
print('On average, tips are highest on',day,'during',meal,'\n')
print(day_meal_avg,'\n')

#q3
print('Q3 ---------------------------------------------------------')
corr = df[['tip_percent','total_bill']].corr().total_bill.tip_percent
print('The correlation between total bill price and tip is',corr,'\n')

#q4
print('Q4 ---------------------------------------------------------')
df_tip_size = df[['tip_percent','size']].rename(index=str, columns={'size': 'group_size'})
corr = df_tip_size.corr().group_size.tip_percent
print('The correlation between group size and tip is',corr,'\n')

#q5
print('Q5 ---------------------------------------------------------')
if df['smoker'].describe()['top'] == 'No':
    nonsmoker = np.round(((len(df) - df['smoker'].describe()['freq']) / len(df))*100,2)
else:
    nonsmoker = np.round((df['smoker'].describe()['freq'] / len(df))*100,2)
print(nonsmoker,'% of patrons are non-smokers\n')

#q8
print('Q8 ---------------------------------------------------------')
df_tip_smoker = df[['tip_percent','smoker']]
df_tip_smoker['smoker'] = df_tip_smoker['smoker'].apply(lambda x: 1 if x == 'Yes' else 0)
corr = df_tip_smoker.corr().smoker.tip_percent
print('The correlation between smokers and tipping is',corr,'\n')

#q9
print('Q9 ---------------------------------------------------------')
day_avg = df.groupby('day')['tip_percent'].mean()
print('The average tip is highest on',day_avg.idxmax(),'\n')
print(day_avg)

#q10
print('Q10 --------------------------------------------------------')
females = df[df.sex.str.contains('Female')]
f_smokers = np.round((len(females[females.smoker.str.contains('Yes')]) / len(females)) * 100, 2)
males =  df[df.sex.str.contains('Male')]
m_smokers = np.round((len(males[males.smoker.str.contains('Yes')]) / len(males)) * 100, 2)
print(f_smokers,'% of females are smokers')
print(m_smokers,'% of males are smokers')
