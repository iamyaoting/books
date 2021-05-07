import os
import sys
import numpy as np
import scipy.io as io
import theano
import theano.tensor as T

sys.path.append('../nn')

from AdamTrainer import AdamTrainer
from AnimationPlot import animation_plot
from network import createcore_torso
from network import createcore_leftleg
from network import createcore_rightleg
from network import createcore_leftarm
from network import createcore_rightarm
from network import create_core

rng = np.random.RandomState(23456)

"""
Xcmu = np.load('../data/processed/data_cmu.npz')['clips']
Xhdm05 = np.load('../data/processed/data_hdm05.npz')['clips']
Xmhad = np.load('../data/processed/data_mhad.npz')['clips']
#Xstyletransfer = np.load('../data/processed/data_styletransfer.npz')['clips']
Xedin_locomotion = np.load('../data/processed/data_edin_locomotion.npz')['clips']
Xedin_xsens = np.load('../data/processed/data_edin_xsens.npz')['clips']
Xedin_misc = np.load('../data/processed/data_edin_misc.npz')['clips']
Xedin_punching = np.load('../data/processed/data_edin_punching.npz')['clips']
"""
#X = np.concatenate([Xcmu, Xhdm05, Xmhad, Xstyletransfer, Xedin_locomotion, Xedin_xsens, Xedin_misc, Xedin_punching], axis=0)

#only train for cmu data
X=np.load('../data/processed/data_cmu.npz')['clips']
#X~(26088,240,73)
#X = np.concatenate([Xcmu, Xhdm05, Xmhad, Xedin_locomotion, Xedin_xsens, Xedin_misc, Xedin_punching], axis=0)
#X~(26088,73,240)
X = np.swapaxes(X, 1, 2).astype(theano.config.floatX)

feet = np.array([12,13,14,15,16,17,24,25,26,27,28,29])

Xmean = X.mean(axis=2).mean(axis=0)[np.newaxis,:,np.newaxis]

#the last element is Xmean[-1]
Xmean[:,-7:-4] = 0.0
Xmean[:,-4:]   = 0.5

Xstd = np.array([[[X.std()]]]).repeat(X.shape[1], axis=1)
#print(Xstd)
Xstd[:,feet]  = 0.9 * Xstd[:,feet]
Xstd[:,-7:-5] = 0.9 * X[:,-7:-5].std()
Xstd[:,-5:-4] = 0.9 * X[:,-5:-4].std()
Xstd[:,-4:]   = 0.5
#print(Xstd.shape)
#save to "preprocess_core.npz"
np.savez_compressed('preprocess_core.npz', Xmean=Xmean, Xstd=Xstd)

X = (X - Xmean) / Xstd
#Disrupt the order of input training samples
I = np.arange(len(X))
rng.shuffle(I); X = X[I]
#X~(26088,73,240)
#print(X.shape)
#E = theano.shared(X, borrow=True)

print("Training Torso:")
XRoot = X[:,0:3,:]
XTorso = X[:,27:39,:]
XTorso = np.concatenate([XRoot,XTorso],axis=1)
print(XTorso.shape)
ETorso = theano.shared(XTorso, borrow=True)
batchsize = 1
network_torso = createcore_torso(rng=rng, batchsize=batchsize, window=XTorso.shape[2])
trainer = AdamTrainer(rng=rng, batchsize=batchsize, epochs=100, alpha=0.00001)
trainer.train(network_torso, ETorso, ETorso, filename='networkcore_torso.npz', logname='log_torso.txt')


print("Traning Leftleg")
XLeftleg = X[:,3:15,:]
print(XLeftleg.shape)
ELeftleg = theano.shared(XLeftleg, borrow=True)
batchsize = 1
network_leftleg = createcore_leftleg(rng=rng, batchsize=batchsize, window=XLeftleg.shape[2])
trainer = AdamTrainer(rng=rng, batchsize=batchsize, epochs=100, alpha=0.00001)
trainer.train(network_leftleg, ELeftleg, ELeftleg, filename='networkcore_leftleg.npz', logname='log_leftleg.txt')

print("Traning Rightleg")
XRightleg = X[:,15:27,:]
print(XRightleg.shape)
ERightleg = theano.shared(XRightleg, borrow=True)
batchsize = 1
network_rightleg = createcore_rightleg(rng=rng, batchsize=batchsize, window=XRightleg.shape[2])
trainer = AdamTrainer(rng=rng, batchsize=batchsize, epochs=100, alpha=0.00001)
trainer.train(network_rightleg, ERightleg, ERightleg, filename='networkcore_rightleg.npz', logname='log_rightleg.txt')

print("Traing Leftarm")
XLeftarm = X[:,39:51,:]
print(XLeftarm.shape)
ELeftarm = theano.shared(XLeftarm, borrow=True)
batchsize = 1
network_leftarm = createcore_leftarm(rng=rng, batchsize=batchsize, window=XLeftarm.shape[2])
trainer = AdamTrainer(rng=rng, batchsize=batchsize, epochs=100, alpha=0.00001)
trainer.train(network_leftarm, ELeftarm, ELeftarm, filename='networkcore_leftarm.npz', logname='log_leftarm.txt')


print("Traing Rightarm")
XRightarm = X[:,51:63,:]
print(XRightarm.shape)
ERightarm = theano.shared(XRightarm, borrow=True)
batchsize = 1
network_rightarm = createcore_rightarm(rng=rng, batchsize=batchsize, window=XRightarm.shape[2])
trainer = AdamTrainer(rng=rng, batchsize=batchsize, epochs=100, alpha=0.00001)
trainer.train(network_rightarm, ERightarm, ERightarm, filename='networkcore_rightarm.npz', logname='log_rightleg.txt')


"""
batchsize = 1
network = create_core(rng=rng, batchsize=batchsize, window=X.shape[2])

trainer = AdamTrainer(rng=rng, batchsize=batchsize, epochs=100, alpha=0.00001)
#input=E
#output=E
#save to "network_core.npz"
trainer.train(network, E, E, filename='network_core.npz')
"""





