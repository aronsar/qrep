from Wrapper.layers import *
from Wrapper.wrappers import make_atari, wrap_deepmind, wrap_pytorch
import math, random
import gym
import numpy as np
from os import path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
USE_CUDA = torch.cuda.is_available()
from dqn import QLearner, ReplayBuffer
import argparse 

parser = argparse.ArgumentParser() 
parser.add_argument(
    '--num_frames',
    '--nf',
    type=int,
    default=1000,
    help="The number of frames to analyze")
parser.add_argument(
    '--loadpath',
    '--lp',
    type=str, 
    help="Path from which to load pretrained model.") 
args = parser.parse_args()


# Initialize environment
env_id = "PongNoFrameskip-v4"
env = make_atari(env_id)
env = wrap_deepmind(env)
env = wrap_pytorch(env)
state = env.reset()


# Model initialization
replay_buffer = ReplayBuffer(100000)
model        = QLearner(env, 1000000, 32, .99, replay_buffer)


# Model loading
if args.loadpath and path.exists(args.loadpath):
    print("Loading model from " + args.loadpath)
    model.load_state_dict(torch.load(args.loadpath, map_location='cpu'))


# Setting up gpu
if USE_CUDA:
    model = model.cuda()
    print("Using cuda...")


def generate_filename(directory, frame_idx, args, file_ending):
    filename = str(int(frame_idx / 1000)) + "k_" + \
               str(args.learning_rate) + "_" + \
               str(args.gamma) + "_" + \
               str(int(args.replay_buffer / 1000)) + "k_" + \
               str(int(args.copy_frequency / 1000)) + "k_" + \
               str(int(args.epsilon_decay/1000)) + file_ending
    filepath = path.join(directory, filename)
    if path.exists(filepath):
        version_num = 2
        while path.exists(filepath + ".v" + str(version_num)):
            version_num += 1
        chkpt_savepath = filepath + ".v" + str(version_num)
    return filepath


# Setting up data structures
losses = []
all_rewards = []
episode_reward = 0


# Doing some frames
for frame_idx in range(1, args.num_frames + 1):
    epsilon = 0

    # Perform an action based on current state
    action = model.act(state, epsilon)
    next_state, reward, done, _ = env.step(action)

    # Push <s, a, r, s'> t the replay buffer
    replay_buffer.push(state, action, reward, next_state, done)

    # store the new state and the cummulative reward for the game
    state = next_state
    episode_reward += reward

    
    if done: # game is over
        state = env.reset()
        all_rewards.append((frame_idx, episode_reward))
        episode_reward = 0


bufferlist = list(replay_buffer.buffer)
states = np.array([bufferlist[x][0] for x in range(len(bufferlist))])
actions = np.array([bufferlist[x][1] for x in range(len(bufferlist))])
reward = np.array([bufferlist[x][1] for x in range(len(bufferlist))])

import pdb; pdb.set_trace()

# one paddle has 147 in it
# another has 148
# and the ball has one number >200, but 236 is banned

# convert each state into a 3D energy map (one for each paddle and the ball)
    # for each white pixel, add cone of distances to zero matrix
    # now an elementwise "euclidean" distance makes sense

from sklearn.manifold import LocallyLinearEmbedding

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
X_reduced = ll.fit_transform(X) # X is a matrix of [num_states, state_dim]

import matplotlib.pyplot as plt

plt.scatter(X_reduced[:,0], X_reduced[:,1], c=labels, alpha=0.5)
plt.show()
