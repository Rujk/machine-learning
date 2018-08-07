# -*- coding: utf-8 -*-
"""
Simple Linear Regression

This file contains the Python version of code for Prof Andrew Ng's course Machine Learning - Week 1 - ex1 program -
one variable Linear Regression.

Data is contained in a csv file.
"""

import numpy as np
from common_utilities import *

# load data
dataset = np.loadtxt('ex1data1.csv', delimiter=',', skiprows=1)

# split dataset into X and Y (input and labels)
X = dataset[:,0:1]
Y = dataset[:,1]
Y = Y.reshape((Y.size,1)) 
m = Y.size

# add a column of ones to X as the intercept column (first column)
ones = np.ones((X.shape[0],1))
X = np.insert(X,0,1, axis=1) 

# initialize weights, number of iterations and the learning rate
W = np.zeros((1,2))
iterations=1500
alpha=0.01

# compute cost at iteration zero for validating the correctness of code
cost = compute_cost(X, Y, W)
print("cost at iteration 0=", cost)

# now train the neural network by using gradient descent
W, J_history = batchGradientDescent(X, Y, W, alpha, iterations)

# print final values of cost and the weights
print("final cost=", J_history[-1]) 
print("final weights=", W)

# now make predictions with the weights determined by gradient descent
# prediction for area with 35,000 population
prediction_1 = h(W,[1,3.5]) * 10000
print("prediction_1=", prediction_1)
# prediction for area with 70,000 population
prediction_2 = h(W,[1,7]) * 10000
print("prediction_2=", prediction_2)
