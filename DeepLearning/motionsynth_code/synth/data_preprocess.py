# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 14:50:25 2018

@author: Sissi_萤
"""

import os
import sys
import numpy as np
import scipy.io as io
import theano
import theano.tensor as T

sys.path.append('../nn')

# X~(17944,240,73)
X = np.load('../data/processed/data_cmu.npz')['clips']

# X~(17944,73,240)
X = np.swapaxes(X, 1, 2).astype(theano.config.floatX)
preprocess = np.load('preprocess_core.npz')
X = (X - preprocess['Xmean']) / preprocess['Xstd']

"""
---from "export.py"
positions = global_positions[:,np.array([
         0,
         2,  3,  4,  5,
         7,  8,  9, 10,
        12, 13, 15, 16,
        18, 19, 20, 22,
        25, 26, 27, 29])]

Torso={0,12,13,15,16}
LeftLeg={2,3,4,5}
RightLeg={7,8,9,10}
LeftArm={18,19,20,22}
RightArm={25,26,27,29}

"""
Xcmu=X[:,0:63,:]

for i in range(len(X)):
    Xcmu[i,(0*3):((0+1)*3-1),:]

    
