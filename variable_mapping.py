# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 14:40:56 2020

@author: smail
"""

#map letter to index

def generate_variable_list(alphabet_size,num_variables):
    lower_case = []
    for i in range(26):
        lower_case.append(chr(i+97))
    
    variables = lower_case[:alphabet_size]
    prefix_index = 0
    letter_index = 0
    prefix = variables[0]
    for i in range(num_variables):
        if ((i+1)) % alphabet_size == 0:   
            variables.append(prefix+lower_case[letter_index])
            prefix_index += 1
            prefix = variables[prefix_index]
            letter_index =0
            
        else:
            variables.append(prefix+lower_case[letter_index])
            letter_index +=1
            
    return variables

alphabet_size = 26
num_variables = 80
variables = generate_variable_list(alphabet_size,num_variables)
print(variables)
            
    # for item in variables:
    #     print(item)

# print(variables[count]+lower_case[letter_index]
    
# def input_output_map(**kwargs):
#     output = kwargs["x"] + kwargs["y"]
#     return output

# from inspect import signature
# x_dim = len(signature(input_output_map).parameters)
# print(x_dim)
    
# dict_ = {"x":1,
#          "y":2}
# temp = input_output_map(**dict_)
# print(temp)