# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 16:20:22 2021

@author: jubil
"""
import numpy as np


from test_mlp import test_mlp, STUDENT_NAME, STUDENT_ID
from acc_calc import accuracy 


y_pred = test_mlp("X_val.csv")

test_labels = np.loadtxt(open("y_val.csv", "rb"), delimiter=",", skiprows=0)

test_accuracy = accuracy(test_labels, y_pred)*100