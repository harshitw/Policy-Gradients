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
prev_x = None # used in computing difference frame
xs, hs, dlogps, drs = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0

while True:
    if render: env.render()
    # preprocess the observation, set input to the network to be difference image
    curr_x = prepro(observation)
    x = curr_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = curr_x

    # forward the policy network and sample an action from the returned policy
    aprob, h = policy_forward(x)
    action = 2 if np.random.uniform() < aprob else 3 # roll the dice!

    # record various intermediate (need for later backprop)
    xs.append(x)# observation
    hs.append(h)# hidden state
    y = 1 if action == 2 else 0 # fake label
    dlogps.append(y - aprob) # a grad that encourages the action that was taken to be taken

    # step the environment and get new results
    observation, reward, done, info = env.step(action)
    reward_sum += reward

    drs.append(reward)
    if done: # if an episode is finished
        episode_number += 1
        # stack together all inputs, hidden state, action gradients and reward for this episode
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        xs, hs, dlogp, drs = [],[],[],[] # reset array memory

        # compute the discounted reward backwards through the time
        discounted_epr = discount_rewards(epr)
        # normalizing the rewards
        discounted_epr = (discounted_epr - np.mean(discounted_epr)) / np.std(discounted_epr)

        epdlogp *= discounted_epr # modulate the gradient with advantage (policy gradient magic happens right here)
        grad = policy_backward(eph, epdlogp)
        for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch

        # perform rmsprop parameter update every batch size episodes
        if episode_number % batch_size == 0:
            for k, v in model.iteritems():
                g = grad_buffer[k] # gradient
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v) # reset the gradient buffer

        running_reward = reward_sum if running_reward is None else running_reward*0.99 + running_sum*0.01
        if episode_number % 100 == 0:
            pickle.dump(model, open('save.p', 'wb'))
        reward_sum = 0
        observation = env.reset()
        prev_x = None
    if reward != 0: # pong has reward 1 or -1 when the game ends
        print('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else '!!!!!!')
