# Train an agent to play a game of pong using stochastic policy gradient
import numpy as np
import cPickle as pickle
import gym

# hyperparameters
H = 200 # no of hidden neurons
batch_size = 10 # every how many episode do a parameter update
learning_rate = 1e-4
gamma = 0.99
decay_rate = 0.99 # decay factor of RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint
render = False

# model intitialization
D = 80*80 # input dimensionality: 80*80 grid
if resume:
    model = pickle.load(open('save.p', 'rb'))
else:
    model = {}
    model['W1'] = np.random.randn(H, D) / np.sqrt(D) # "Xavier inititlization"
    model['W2'] = np.random.randn(H) / np.sqrt(H)

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def prepro(I):
    """ prepro 210*120 uint8 frame into 80*80 1D float vector"""
    I = I[35:195] # crop
    I = I[::2, ::2, 0] # downsample by a factor of 2
    I[I == 144] = 0 # erase background type 1
    I[I == 109] = 0 # erase background type 2
    I[I != 0] = 1 # everything else (paddles, balls) set to 1
    return I.astype(np.float).ravel()

def discount_rewards(r):
    """take 1D float array of rewards and compute the discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        if r[t] != 0:
            running_add = 0 # reset the sum since this is game boundary (pong specific)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def policy_forward(x):
    h = np.dot(model['W1'], x)
    h[h<0] = 0 # ReLU non linearity
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    return p, h # re

def policy_backward(eph, epdlogp):
    """ backward pass. (eph is an array of intermidiate hidden states)"""
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2']) # used for multiplication of n*1 and 1*m vectors
    dh[eph <= 0] = 0 # backprop prelu
    dW1 = np.dot(dh.T, epx)
    return {'W1': dW1, 'W2':dW2}

env = gym.make("Pong-v0")
observation = env.reset()
