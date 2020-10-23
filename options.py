# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:54:34 2020

@author: smail
"""

config = {"use_rescaled_MSE":True,
          "epochs1":100,
          "epochs2":100,
          "threshold_value":0.01,
          "a_L_0.5":5e-3,
          "use_phase2":True,
          "use_thresholding_before_phase2":True,
          "lambda_reg":5e-3,
          "batch_size":200,
          "phase1_lr":1e-4,
          "phase2_lr":1e-5,
          "eql_number_layers":2,
          "optimizer":"rmsprop",
          "use_regularization_phase2":True,
          "number_trials":1,
          "steps_ahead":6,
          "phase2_from_file":False,
          "non_masked_weight_file":None,
          "type_loss":"l12_smooth"
          }

#eql functions
