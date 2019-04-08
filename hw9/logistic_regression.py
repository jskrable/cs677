#!/usr/bin/env python3
# coding: utf-8
"""
logistic_regression.py
04-07-19
jack skrable
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


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


# Function to plot logistic regression line and drawn line on sep. dataset
def logistic_separation(df):

	# Scale data
	print('Preprocessing...')
	X = df[['Std','Mean']].values
	scaler = StandardScaler()
	scaler.fit(X)
	X = scaler.transform(X)
	y = df['Score'].values

	# Train model
	print('Training regression model...')
	lrc = LogisticRegression(solver='lbfgs')
	lrc.fit(X,y)

	# Get graph data
	print('Plotting graphs...')
	xx, yy = np.hsplit(X,2)
	x_range = np.linspace(xx.min(), xx.max())
	y_drawn = [0.7 * x + 0.01 for x in x_range]
	y_log = [lrc.coef_[0][1] * x + lrc.intercept_[0] for x in x_range]

	# Plot graphs
	sns.scatterplot(xx.ravel(), yy.ravel(), hue=y, palette={1:'olivedrab',0:'firebrick'})
	sns.lineplot(x_range, y_drawn, label='drawn')
	sns.lineplot(x_range, y_log, label='regression')

	# Add labels
	plt.xlabel('Mean Weekly Return')
	plt.ylabel('Std Deviation')
	plt.title('Scaled Mean Return vs. Std. Deviation: Logistic Regression Line')
	plt.show()

	return 0

# Function to predict 2018 labels using logistic regression
def prediction(df):

	print('Preprocessing data...')
	X = df[['Mean','Std']].values
	# Get train and test sets by year
	X_train = df.loc['2017'][['Mean','Std']].values
	y_train = df.loc['2017']['Score'].values
	X_test = df.loc['2018'][['Mean','Std']].values
	y_test = df.loc['2018']['Score'].values
	# Train scaler
	scaler = StandardScaler()
	scaler.fit(X)
	# Scale datasets
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)

	# Train model
	print('Training model...')
	lrc = LogisticRegression(solver='lbfgs')
	lrc.fit(X_train, y_train)

	# Evaluate model
	print('Testing model...')
	prediction = lrc.predict(X_test)
	accuracy = np.mean(prediction == y_test)

	return accuracy



# MAIN
###############################################################################

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

print('Q3: Logistic Regression Plot------------------------------------------')
print('Cleaning data...')
cleaned = weekly[weekly.apply(remove_points, axis=1)]
logistic_separation(cleaned)

print('Q4: Logistic Regression Classifier------------------------------------')
accuracy = prediction(weekly)
print('Accuracy:', np.round(accuracy * 100, 2), '%')


	
