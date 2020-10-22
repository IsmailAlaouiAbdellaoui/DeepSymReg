# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:54:34 2020

@author: smail
"""

config = {"use_rescaled_MSE":False,
          "epochs1":3,
          "epochs2":3,
          "threshold_value":0.01,
          "a_L_0.5":5e-2,
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
