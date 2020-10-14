# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 12:59:29 2020

@author: smail
"""

import numpy as np


    
def get_dataset():
    total_time_steps = 2000
    cities = 5
    features = 5
    lags = 2
    tensor_input = np.random.rand(total_time_steps//lags,features,cities,lags)
    
    output_data = np.zeros((total_time_steps//lags,1))
    for i in range(total_time_steps//lags):
        output_data[i]= tensor_input[i][0][0][0]**2+tensor_input[i][0][0][1]**2
        
    return tensor_input,output_data
    
    
# def test_train_data(index_output):
#     value_out = output_data[index_output]
#     first_value = tensor_input[index_output][0][0][0]
#     second_value = tensor_input[index_output][0][0][1]
#     assert value_out == first_value**2 +  second_value**2,print("not equal !")
    

    

# test_train_data(998)
# for i in range(10):
#     print(output_data[i])


# m1_12_vars = [6.28E-03,7.67E-03,3.33E-01,2.76E-02]
#     m1_24_vars = [9.74E-03,1.07E-02,4.08E-01,3.80E-02]
#     m1_48_vars = [1.50E-02,2.34E-02,6.18E-01,5.86E-02]
#     m1_72_vars = [2.89E-02,4.09E-02,6.70E-01,1.03E-01]
#     m1_list = [m1_12_vars,m1_24_vars,m1_48_vars,m1_72_vars]
    
#     m2_12_vars = [3.74E-02,1.28E-01,4.86E-01,1.96E-01]
#     m2_24_vars = [2.42E-02,6.46E-02,6.42E-01,1.14E-01]
#     m2_48_vars = [2.52E-02,2.42E-02,6.36E-01,7.81E-02]
#     m2_72_vars = [3.66E-02,8.42E-02,9.74E-01,1.26E-01]
#     m2_list = [m2_12_vars,m2_24_vars,m2_48_vars,m2_72_vars]
    
#     m3_12_vars = [3.27E-02,1.23E-01,4.93E-01,1.89E-01]
#     m3_24_vars = [1.97E-02,6.69E-02,5.09E-01,9.43E-02]
#     m3_48_vars = [1.72E-02,2.24E-02,5.36E-01,5.93E-02]
#     m3_72_vars = [2.49E-02,6.29E-02,5.95E-01,1.08E-01]
#     m3_list = [m3_12_vars,m3_24_vars,m3_48_vars,m3_72_vars]
    
#     m4_12_vars = [5.57E-03,6.64E-03,2.94E-01,2.25E-02]
#     m4_24_vars = [8.17E-03,1.02E-02,3.86E-01,3.42E-02]
#     m4_48_vars = [1.23E-02,2.07E-02,5.77E-01,5.42E-02]
#     m4_72_vars = [1.66E-02,3.88E-02,6.69E-01,9.71E-02]
    
    
    