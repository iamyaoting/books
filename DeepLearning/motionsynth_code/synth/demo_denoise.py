import sys
import numpy as np
import scipy.io as io
import os    
os.environ['THEANO_FLAGS'] = "device=cpu"  
import theano
import theano.tensor as T

sys.path.append('../nn')

from network import createcore_torso
from network import createcore_leftleg
from network import createcore_rightleg
from network import createcore_leftarm
from network import createcore_rightarm
from network import create_core
from constraints import constrain, foot_sliding, joint_lengths, trajectory, multiconstraint

#the starting point for a sequence of pseudorandom number
rng = np.random.RandomState(23455)

#X = np.load('../data/processed/data_edin_locomotion.npz')['clips']
#X = np.load('../data/processed/data_hdm05.npz')['clips']

# X~(17944,240,73)
X = np.load('../data/processed/data_hdm05.npz')['clips']
# X~(17944,73,240)
X = np.swapaxes(X, 1, 2).astype(theano.config.floatX)

preprocess = np.load('preprocess_core.npz')
X = (X - preprocess['Xmean']) / preprocess['Xstd']

#print(preprocess['Xmean'])

batchsize = 1
#window=240
window = X.shape[2]

X = theano.shared(X, borrow=True)

networkcore_torso = createcore_torso(batchsize=batchsize, window=window, dropout=0.0, depooler=lambda x,**kw: x/2)
networkcore_torso.load(np.load('networkcore_torso.npz'))
networkcore_leftarm = createcore_leftarm(batchsize=batchsize, window=window, dropout=0.0, depooler=lambda x,**kw: x/2)
networkcore_leftarm.load(np.load('networkcore_leftarm.npz'))
networkcore_rightarm = createcore_rightarm(batchsize=batchsize, window=window, dropout=0.0, depooler=lambda x,**kw: x/2)
networkcore_rightarm.load(np.load('networkcore_rightarm.npz'))
networkcore_leftleg = createcore_rightarm(batchsize=batchsize, window=window, dropout=0.0, depooler=lambda x,**kw: x/2)
networkcore_leftleg.load(np.load('networkcore_leftleg.npz'))
networkcore_rightleg = createcore_rightarm(batchsize=batchsize, window=window, dropout=0.0, depooler=lambda x,**kw: x/2)
networkcore_rightleg.load(np.load('networkcore_rightleg.npz'))

network = create_core(batchsize=batchsize, window=window, dropout=0.0, depooler=lambda x,**kw: x/2)
network.load(np.load('network_core.npz'))

from AnimationPlot import animation_plot

# 2021
# 13283

for _ in range(10):
    #[0，17944)
    #index = rng.randint(X.shape[0].eval())
    index = 0
    print(index)
    # Xorgi~(1,73,240)
    Xorgi = np.array(X[index:index+1].eval())
    #print(Xorgi.shape)
    #extract torso part
    Xorgi_Root = Xorgi[:,0:3,:]
    Xorgi_Torso = Xorgi[:,27:39,:]
    Xorgi_Torso = np.concatenate([Xorgi_Root,Xorgi_Torso],axis=1)
    Xnois_Torso = ((Xorgi_Torso * rng.binomial(size=Xorgi_Torso.shape, n=1, p=0.5)) / 0.5).astype(theano.config.floatX)
    #Xnois_Torso = (Xorgi_Torso + 0.05*np.random.randn(1,15,240)+0.1).astype(theano.config.floatX)
    Xrecn_Torso = np.array(networkcore_torso(Xnois_Torso).eval())
    #print(Xrecn_Torso.shape)
    
    #extract leftleg part
    Xorgi_Leftleg = Xorgi[:,3:15,:]
    Xnois_Leftleg = ((Xorgi_Leftleg * rng.binomial(size=Xorgi_Leftleg.shape, n=1, p=0.5)) / 0.5).astype(theano.config.floatX)
    #Xnois_Leftleg = (Xorgi_Leftleg + 0.05*np.random.randn(1,12,240)+0.1).astype(theano.config.floatX)
    Xrecn_Leftleg = np.array(networkcore_leftleg(Xnois_Leftleg).eval())
    
    #extract rightleg part
    Xorgi_Rightleg = Xorgi[:,15:27,:]
    Xnois_Rightleg = ((Xorgi_Rightleg * rng.binomial(size=Xorgi_Rightleg.shape, n=1, p=0.5)) / 0.5).astype(theano.config.floatX)
    #Xnois_Rightleg = (Xorgi_Rightleg + 0.05*np.random.randn(1,12,240)+0.1).astype(theano.config.floatX)
    Xrecn_Rightleg = np.array(networkcore_rightleg(Xnois_Rightleg).eval())
    
    #extract leftarm part
    Xorgi_Leftarm = Xorgi[:,39:51,:]
    Xnois_Leftarm = ((Xorgi_Leftarm * rng.binomial(size=Xorgi_Leftarm.shape, n=1, p=0.5)) / 0.5).astype(theano.config.floatX)
    #Xnois_Leftarm = (Xorgi_Leftarm + 0.05*np.random.randn(1,12,240)+0.1).astype(theano.config.floatX)
    Xrecn_Leftarm = np.array(networkcore_leftarm(Xnois_Leftarm).eval())
    
    #extract rightarm part
    Xorgi_Rightarm = Xorgi[:,51:63,:]
    Xnois_Rightarm = ((Xorgi_Rightarm * rng.binomial(size=Xorgi_Rightarm.shape, n=1, p=0.5)) / 0.5).astype(theano.config.floatX)
    #Xnois_Rightarm = (Xorgi_Rightarm + 0.05*np.random.randn(1,12,240)+0.1).astype(theano.config.floatX)
    Xrecn_Rightarm = np.array(networkcore_rightarm(Xnois_Rightarm).eval())
    
    Xnois = np.concatenate([Xnois_Torso[:,0:3,:], Xnois_Leftleg, Xnois_Rightleg, Xnois_Torso[:,3:15,:], Xnois_Leftarm, Xnois_Rightarm, Xorgi[:,63:73,:]],axis=1)
    print(Xnois.shape)
    Xrecn = np.concatenate([Xrecn_Torso[:,0:3,:], Xrecn_Leftleg, Xrecn_Rightleg, Xrecn_Torso[:,3:15,:], Xrecn_Leftarm, Xrecn_Rightarm, Xorgi[:,63:73,:]], axis=1)
    
    
    Xorgi = (Xorgi * preprocess['Xstd']) + preprocess['Xmean']
    Xnois = (Xnois * preprocess['Xstd']) + preprocess['Xmean']
    Xrecn = (Xrecn * preprocess['Xstd']) + preprocess['Xmean']
    
    np.save("./denoise/Xorgi.npy",Xorgi)
    np.save("./denoise/Xnois.npy",Xnois)
    
    print(Xorgi[:,-7:-4].shape)
    # H'=argmin[Pos(H)+Bone(H)+Traj(H)]
    Xrecn = constrain(Xrecn, network[0], network[1], preprocess, multiconstraint(
        foot_sliding(Xorgi[:,-4:].copy()),#foot sliding information(-4,-3,-2,-1)
        joint_lengths(),
        trajectory(Xorgi[:,-7:-4])), alpha=0.01, iterations=50)#input trajectory(-7,-6,-5)
    
    Xrecn[:,-7:-4] = Xorgi[:,-7:-4]
    
    np.save("./denoise/Xrecn.npy",Xrecn)
    print("construction done")
    
    animation_plot([Xnois, Xrecn, Xorgi], interval=15.15)
    
    """
    Xnois = ((Xorgi * rng.binomial(size=Xorgi.shape, n=1, p=0.5)) / 0.5).astype(theano.config.floatX)
    #Xnois = (Xorgi + 0.05*np.random.randn(1,73,240)+0.1).astype(theano.config.floatX)
    Xrecn = np.array(network(Xnois).eval())    
    #print(Xrecn.shape)
    """
    """
    Xorgi = (Xorgi * preprocess['Xstd']) + preprocess['Xmean']
    Xnois = (Xnois * preprocess['Xstd']) + preprocess['Xmean']
    Xrecn = (Xrecn * preprocess['Xstd']) + preprocess['Xmean']
    
    np.save("./denoise/Xorgi.npy",Xorgi)
    np.save("./denoise/Xnois.npy",Xnois)
    
    Xrecn = constrain(Xrecn, network[0], network[1], preprocess, multiconstraint(
        foot_sliding(Xorgi[:,-4:].copy()),
        joint_lengths(),
        trajectory(Xorgi[:,-7:-4])), alpha=0.01, iterations=50)
    
    Xrecn[:,-7:-4] = Xorgi[:,-7:-4]
    
    np.save("./denoise/Xrecn.npy",Xrecn)
    print("construction done")
    print("construction done")
    print("construction done")
    print("construction done")
    print("construction done")
    print("construction done")
    print("construction done")
    print("construction done")
    print("construction done")
    print("construction done")
    print("construction done")
    print("construction done")
    animation_plot([Xnois, Xrecn, Xorgi], interval=15.15)
    """    