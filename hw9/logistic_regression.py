#!/usr/bin/env python3
# coding: utf-8
"""
naive_bayes.py
03-31-19
jack skrable
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ticker = 'SYK'

# Get path of set
if os.name == 'posix':
    input_dir = r'/home/jskrable/code/cs677/datasets'
else:
    input_dir = r'C:\Users\jskrable\code\cs677\datasets'
output_file = os.path.join(input_dir, ticker + '_labelled.csv')

# Read labelled file into pandas dataframe
print('Reading data file...')
df = pd.read_csv(output_file)

# Get weekly summary dataframe
print('Applying group functions...')
df = df.loc[df.Year.isin([2017,2018])]
df['Date'] = pd.to_datetime(df['Date']) - pd.to_timedelta(7, unit='d')
group = df.groupby([pd.Grouper(key='Date', freq='W')])
weekly = group['Return'].mean().to_frame('Mean')
weekly['Std'] = group['Return'].std()
weekly['Score'] = group['Score'].first()

# Extract data for classification
print('Preprocessing data...')
# Fill NaNs
weekly = weekly.fillna(0)
# weekly = weekly.reset_index()

# Function to return False for bad data points
def remove_points(row):
	# Calculate linear guess
	slope = 0.7 * row.Mean + 0.01

	if slope < row.Std and row.Score == 0:
		return True
	elif slope > row.Std and row.Score ==1:
		return True
	else:
		return False


# Function to plot both graphs
def plot_points(x, y, c, title):
	plt.scatter(x, y, color=c)
	x_line = np.linspace(min(x),max(x)+0.01)
	y_line_1 = [0.7*x + 0.01 for x in x_line]
	# y_line_2 = 
	plt.plot(x_line, y_line_1, '--')
	plt.xlabel('Mean Weekly Return')
	plt.ylabel('Std Deviation')
	plt.title(title)
	plt.show()


# Plot initial graph with simple linear separator
print('Plotting inital graph...')
x = [x for x in weekly.Mean]
y = [y for y in weekly.Std]
c = ['green' if x > 0 else 'red' for x in weekly.Score]
t = 'Best-Guess Linear Separator'
plot_points(x,y,c,t)

# Create cleaned df
print('Cleaning data...')
cleaned = weekly[weekly.apply(remove_points, axis=1)]

# Plot cleaned graph with simple linear separator
print('Plotting cleaned graph...')
x = [x for x in cleaned.Mean]
y = [y for y in cleaned.Std]
c = ['green' if x > 0 else 'red' for x in cleaned.Score]
t = 'Cleaned Plot'
plot_points(x,y,c,t)

X = cleaned[['Std','Mean']].values
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
y = cleaned['Score'].values

lrc = LogisticRegression(solver='lbfgs')
lrc.fit(X,y)

sns.regplot(x='Mean',y='Std',data=cleaned,logistic=True)     


def prediction(df):

	print('Preprocessing data...')
	X = df[['Mean','Std']].values
	X_train = df.loc['2017'][['Mean','Std']].values
	y_train = df.loc['2017']['Score'].values
	X_test = df.loc['2018'][['Mean','Std']].values
	y_test = df.loc['2018']['Score'].values
	scaler = StandardScaler()
	scaler.fit(X)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)

	print('Training model...')
	lrc = LogisticRegression(solver='lbfgs')
	lrc.fit(X_train, y_train)

	print('Testing model...')
	prediction = lrc.predict(X_test)
	accuracy = np.mean(prediction == y_test)

	return accuracy

# MAIN
###############################################################################


print('Q4: Logistic Regression Classifier------------------------------------')
accuracy = prediction(weekly)
print('Accuracy:', np.round(accuracy * 100, 2), '%')


	
