# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 23:43:22 2020

@author: smail
"""

from scipy.io import loadmat
filename = "scale1.mat"
# filename = "scale2.mat"
# filename = "step1.mat"
# filename = "step2.mat"
data = loadmat('Denmark_data/{}'.format(filename))
print(data.keys())
print(data["y_max_tr"].shape)
# print(data["__globals__"])
# print(data["Xtr"].shape)
# temp = data["Xtr"]
# temp = temp.reshape((temp.shape[0],80))
# print(temp.shape)
# temp2 = data["Ytr"][:,0]
# temp2 = temp2.reshape(temp2.shape[0],1)
# temp3 = data["Xtest"]
# print(temp3.shape)
# print(86761//200)
