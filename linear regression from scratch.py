#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 17:18:17 2020

@author: feichang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,0].values
Y = dataset.iloc[:,1].values

#normalize the data to [-1,1]. in this case, [0,1] since it's all positive
norm_X = max(X, key = abs)
norm_Y = max(Y, key = abs)
X /= norm_X
Y/= norm_Y

#initialize variables for gradient descent
beta = [1,1]
# model is y = ax + b, or beta_0 + beta_1*x
alpha = 0.2
# learning rate
tol = 0.1
#tolerence

def compute_grad(beta, x , y):
    grad = [0,0]
    #reminder to my rookie self, make sure the 2 is float
    # I did the gradient math my self
    grad[0] = 2. * np.mean(beta[0] + beta[1] * x - y)
    grad[1] = 2. * np.mean(x * (beta[0] + beta[1] * x - y))
    return np.array(grad)

def update_beta(beta, alpha, grad):
    return np.array(beta) - alpha * grad

def mean_square_error(beta, x, y):
    # create an array of square errors
    err = (beta[0] + beta[1] * x - y) ** 2
    # mean square error
    mse = np.sqrt(np.mean(err))
    return mse

#define the variables, calculate for the first time
grad = compute_grad(beta, X, Y)
loss_func = mean_square_error(beta, X, Y)
beta = update_beta(beta, alpha, grad)
loss_new = mean_square_error(beta, X, Y)

gen = 1
