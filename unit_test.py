# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 15:37:44 2020

@author: smail
"""

DOMAIN_TEST = (-2, 2) 
DOMAIN = (-1, 1)
from inspect import signature
import numpy as np
NOISE_SD = 0 

def generate_data(func, N, range_min=DOMAIN[0], range_max=DOMAIN[1]):
    """Generates datasets."""
    x_dim = len(signature(func).parameters)     # Number of inputs to the function, or, dimensionality of x
    x = (range_max - range_min) * np.random.random([N, x_dim]) + range_min
    y = np.random.normal([[func(*x_i)] for x_i in x], NOISE_SD)
    return x, y

x,y = generate_data(lambda x: x,256)
# print(type(x))
print(x.shape)
print(type(y))