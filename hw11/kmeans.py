#!/usr/bin/env python3
# coding: utf-8
"""
kmeans.py
04-21-19
jack skrable
"""

import os
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ticker = 'SYK'
label = True

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
df['Date'] = pd.to_datetime(df['Date']) - pd.to_timedelta(7, unit='d')
group = df.groupby([pd.Grouper(key='Date', freq='W')])
weekly = group['Return'].mean().to_frame('Mean')
weekly['Std'] = group['Return'].std()
weekly['Score'] = group['Score'].first()

# Fill NaNs
weekly = weekly.fillna(0)

# Split to training and testing data
weekly = weekly.reset_index()
weekly['Year'] = weekly['Date'].dt.year
test = weekly.loc[weekly['Year'].isin([2018])]
train = weekly.loc[weekly['Year'].isin([2017])]

# Extract data for classification
print('Preprocessing data...')
x = weekly[['Mean', 'Std']].values
y = weekly[['Score']].values
# x_train = train[['Mean', 'Std']].values
# y_train = train['Score'].values
# x_test = test[['Mean', 'Std']].values
# y_test = test['Score'].values

# Scale data
scaler = StandardScaler()
scaler.fit(x)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
x = scaler.transform(x)


def kmeans(X, k):
    # Config kmeans
    km = KMeans(n_clusters=k,
                init='random',
                n_init=10,
                max_iter=300,
                random_state=0)

    km.fit(X)
    return km


def simple_5(X):

    # Plot all points
    print('Plotting all points...')
    sns.scatterplot(X[:,0], X[:,1])
    plt.grid()
    plt.xlabel('Mean Return')  
    plt.ylabel('Std Deviation')
    plt.title('2017 - 2018: All Points')
    plt.show()

    print('Clustering...')
    # Config kmeans
    km = kmeans(X, 5)

    # Get clusters 
    y_km = km.predict(X)

    print('Plotting clusters...')
    # Plot clusters
    pal = sns.hls_palette(n_colors=km.n_clusters, l=0.6)
    sns.scatterplot(X[:,0], X[:,1], hue=y_km, palette=pal)

    # Plot centroids
    pal = sns.hls_palette(n_colors=km.n_clusters, l=0.4)
    sns.scatterplot(km.cluster_centers_[:,0], km.cluster_centers_[:,1], s=250,
                    marker='*', hue=[i for i in range(len(km.cluster_centers_))],
                    palette=pal, legend=False)
    plt.grid()
    plt.xlabel('Mean Return')  
    plt.ylabel('Std Deviation')
    plt.title('2017 - 2018: 5 Clusters')
    plt.show()

def find_best_k(X):

    print('Trying different K-values...')
    results = {k: kmeans(X,k).inertia_ for k in range(1,15)}
    print('Plotting distortion...')
    sns.lineplot(list(results.keys()), list(results.values()),
                 marker='o')
    plt.xlabel('K-value')
    plt.ylabel('Distortion')
    plt.title('Finding Optimal K-value')
    plt.grid()
    plt.show()


def examine_best_k(X, y):

    print('Clustering...')
    # Config kmeans
    km = kmeans(X, 7)
    y_km = km.predict(X)
    print('Processing...')
    df = pd.DataFrame({'mean':X[:,0],'std':X[:,1],'cluster':y_km,'label':y.ravel()})
    ct = df.groupby(['cluster','label']).label.count()
    cs = ct / df.groupby(['cluster']).label.count()

    print('Displaying label distibution by cluster:')
    for g,p in cs.iteritems():
        print('Cluster',g[0],'Label:',g[1],np.round(p*100),'%') 


def custom_kmeans(df, k, o):

    def assignment(df, centroids, o=2):
        for i in centroids.keys():
                # sqrt((x1 - x2)^2 - (y1 - y2)^2)
                df['distance_from_{}'.format(i)] = (
                    (np.abs(df['x'] - centroids[i][0]) ** o + 
                     np.abs(df['y'] - centroids[i][1]) ** o) ** (1/o)
                    )
        centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
        df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
        df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
        # df['color'] = df['closest'].map(lambda x: colmap[x])
        return df


    def update(k):
        for i in centroids.keys():
            centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
            centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
        return k


    def plot(title, df, centroids=None):

        try:
            pal = sns.hls_palette(n_colors=k, l=0.6)
            sns.scatterplot(df.x, df.y, hue=df.closest, palette=pal)

        except AttributeError:
            pal = sns.hls_palette(n_colors=k, l=0.6)
            sns.scatterplot(df.x, df.y, palette=pal)

        # Plot centroids
        if centroids is not None:
            pal = sns.hls_palette(n_colors=k, l=0.4)
            vals = np.array(list(centroids.values()))
            sns.scatterplot(vals[:,0], vals[:,1], s=250,
                            marker='*', hue=[i for i in range(k)],
                            palette=pal, legend=False)
        plt.title(title)
        plt.grid()
        plt.show()


    if o == 1:
        kind = 'Euclidean'
    elif o == 1.5:
        kind = 'Minkowski'
    elif o == 2:
        kind = 'Manhattan'

    np.random.seed(200)
    centroids = {
        i+1: [np.random.uniform(df.x.min(), df.x.max()),
              np.random.uniform(df.y.min(), df.y.max())]
        for i in range(k)
    }


    plot('Initial Centroids: ' + kind, df, centroids)

    df = assignment(df, centroids, o)

    plot('Initial Assignment: ' + kind, df, centroids)    

    old_centroids = copy.deepcopy(centroids)

    centroids = update(centroids)
    
    plot('Initial Update: ' + kind, df, centroids)

    # repeat assignment stage
    df = assignment(df, centroids)

    # Plot results
    plot('Second Assignment: ' + kind, df, centroids)

    # run additional iterations
    # Continue until all assigned categories don't change any more
    i = 0
    while True:
        closest_centroids = df['closest'].copy(deep=True)
        centroids = update(centroids)
        df = assignment(df, centroids)
        i += 1
        if closest_centroids.equals(df['closest']):
            break

    plot('Final Assignments: ' + kind, df, centroids)
    print('Final Group Counts:', kind)
    print(df.groupby(['closest']).count().x)

# MAIN
##########################################################################
print('Q1---------------------------------------------------------------')
simple_5(x)

print('Q2---------------------------------------------------------------')
find_best_k(x)

print('Q3---------------------------------------------------------------')
examine_best_k(x, y)

print('Q4---------------------------------------------------------------')
df1 = pd.DataFrame({'x':x[:,0], 'y':x[:,1]})
custom_kmeans(df1, 3, 1)
custom_kmeans(df1, 3, 1.5)
custom_kmeans(df1, 3, 2)