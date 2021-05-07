# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 14:04:38 2017

@author: Sissi_萤
"""
#used for training the denoise network
import sys
import numpy as np
import scipy.io as io
import os    
os.environ['THEANO_FLAGS'] = "device=cpu"  
import theano
import theano.tensor as T

sys.path.append('../nn')

from network import create_core
from constraints import constrain, foot_sliding, joint_lengths, trajectory, multiconstraint
