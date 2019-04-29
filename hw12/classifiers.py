#!/usr/bin/env python3
# coding: utf-8
"""
random_forest.py
04-29-19
jack skrable
"""

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def support_vector_machine(x_train, y_train, x_test, y_test, kind):
    print('Classifying with SVM...')
    if kind == 'poly':
        svm_classifier = svm.SVC(kernel=kind, degree=2, gamma='auto')
    else:
        svm_classifier = svm.SVC(kernel=kind, gamma='auto')
    svm_classifier.fit(x_train,y_train)
    y_pred = svm_classifier.predict(x_test)
    error_rate = np.mean(y_pred != y_test)
    return error_rate


def logistic_regression(x_train, y_train, x_test, y_test):
    print('Classifying with logistic regression...')
    lrc = LogisticRegression(solver='lbfgs')
    lrc.fit(x_train, y_train)
    y_pred = lrc.predict(x_test)
    error_rate = np.mean(y_pred != y_test)
    return error_rate


def naive_bayes(x_train, y_train, x_test, y_test):
    print('Classifying with naive Bayes...')
    nbc = GaussianNB().fit(x_train, y_train)
    y_pred = nbc.predict(x_test)
    error_rate = np.mean(y_pred != y_test)
    return error_rate


def k_nearest_neighbor(x_train, y_train, x_test, y_test, k):
    print('Classifying with kNN...')
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    error_rate = np.mean(y_pred != y_test)
    return error_rate


def decision_tree(x_train, y_train, x_test, y_test):
    print('Classifying with decision tree...')
    dtc = DecisionTreeClassifier(criterion='entropy')
    dtc.fit(x_train, y_train)
    y_pred = dtc.predict(x_test)
    error_rate = np.mean(y_pred != y_test)
    return error_rate