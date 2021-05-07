# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 16:07:04 2017

@author: Sissi_萤
"""
import numpy as np
import theano.tensor as T
#a = np.load('network_core.npz')

#print(a["L000_L001_W"])
#print(a["L000_L002_b"])
#print(a["L001_L002_W"])
#print(a["L001_L003_b"])

import sys
import numpy as np
import scipy.io as io
import os    
os.environ['THEANO_FLAGS'] = "device=cpu"    
import theano
import theano.tensor as T

sys.path.append('../nn')
#两个点表示上级目录 
#一个点表示当前目录

from network import create_core
from constraints import constrain, foot_sliding, joint_lengths, trajectory, multiconstraint


Xorgi=np.load('./denoise/Xorgi.npy')
#print(b.shape)
Yorgi = np.swapaxes(Xorgi, 1, 2).astype(theano.config.floatX)
#print(b.shape)
np.savetxt("./denoise/Xorgi.txt",Yorgi[0])

Xnois=np.load('./denoise/Xnois.npy')
#print(b.shape)
Ynois = np.swapaxes(Xnois, 1, 2).astype(theano.config.floatX)
#print(b.shape)
np.savetxt("./denoise/Xnois.txt",Ynois[0])


Xrecn=np.load('./denoise/Xrecn.npy')
#print(b.shape)
Yrecn = np.swapaxes(Xrecn, 1, 2).astype(theano.config.floatX)
#print(b.shape)
np.savetxt("./denoise/Xrecn.txt",Yrecn[0])

preprocess = np.load('preprocess_core.npz')

print(preprocess['Xstd'].shape)
print(preprocess['Xmean'].shape)




