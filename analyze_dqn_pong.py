from Wrapper.layers import *
from Wrapper.wrappers import make_atari, wrap_deepmind, wrap_pytorch
import math, random
import gym
import numpy as np
import time
from os import path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
USE_CUDA = torch.cuda.is_available()
from dqn import QLearner, ReplayBuffer
from sklearn.manifold import Isomap, LocallyLinearEmbedding, MDS, SpectralEmbedding, TSNE
import matplotlib.pyplot as plt
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
    default='./models/900k_2e-05_0.995_200k_40k_100.pth',
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
    
    plt.imsave('states/' + str(frame_idx) + '.png', state[0], cmap='gray')
    #env.render()
    #time.sleep(.050)


bufferlist = list(replay_buffer.buffer)
states = np.array([bufferlist[x][0] for x in range(len(bufferlist))])[:,0,0,:,:]
actions = np.array([bufferlist[x][1] for x in range(len(bufferlist))])
reward = np.array([bufferlist[x][1] for x in range(len(bufferlist))])

# one paddle has 147 in it
# another has 148
# and the ball has one number >200, but 236 is banned

# convert each state into a 3D energy map (one for each paddle and the ball)
    # for each white pixel, add cone of distances to zero matrix
    # now an elementwise "euclidean" distance makes sense


def generate_energy_map(indices, state_shape):
    energy_map = np.zeros(state_shape)
    if len(indices) == 0:
        return energy_map

    arange = np.arange(0,84)
    rows = np.repeat(arange[:,np.newaxis], 84, axis=1)
    cols = np.transpose(rows)

    for index in indices:
        row_idx, col_idx = index
        energy_map -= np.sqrt((rows - row_idx)**2 + (cols - col_idx)**2)
    # normalize to between 0 and 1
    energy_map -= np.min(energy_map)
    energy_map /= np.max(energy_map)
    return energy_map



energy_maps = np.zeros(states.shape + (3,))
for state, energy_map in zip(states, energy_maps):
    playing_area = np.zeros(state.shape) # 84 x 84
    playing_area[:, 14:77] = state[:, 14:77] 
    p1_paddle_idxs = np.argwhere(playing_area == 147)
    p2_paddle_idxs = np.argwhere(playing_area == 148)
    ball_idxs  = np.argwhere(playing_area > 200)

    energy_map[:,:,0] = generate_energy_map(p1_paddle_idxs, state.shape)
    energy_map[:,:,1] = generate_energy_map(p2_paddle_idxs, state.shape)
    energy_map[:,:,2] = generate_energy_map(ball_idxs, state.shape)

flattened_energy_maps = energy_maps.reshape(*energy_maps.shape[:1], -1)
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']
action_colors = [colors[action] for action in actions]
reduc_methods = [Isomap, LocallyLinearEmbedding, MDS, SpectralEmbedding, TSNE]

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)


print("Calculating Isomap")
isomap = Isomap(n_neighbors=10)
reduced1 = isomap.fit_transform(flattened_energy_maps)
ax1.scatter(reduced1[:,0], reduced1[:,1], c=action_colors, alpha=.2)
ax1.set_title('Isomap Reduction')

print("Calculating LLE")
lle = LocallyLinearEmbedding(n_neighbors=10)
reduced2 = lle.fit_transform(flattened_energy_maps)
ax2.scatter(reduced2[:,0], reduced2[:,1], c=action_colors, alpha=.2)
ax2.set_title('LLE Reduction')

print("Calculating MDS")
mds = MDS()
reduced3 = mds.fit_transform(flattened_energy_maps)
ax3.scatter(reduced3[:,0], reduced3[:,1], c=action_colors, alpha=.2)
ax3.set_title('MDS Reduction')

print("Calculating tSNE")
tsne = TSNE()
reduced4 = tsne.fit_transform(flattened_energy_maps)
ax4.scatter(reduced4[:,0], reduced4[:,1], c=action_colors, alpha=.2)
ax4.set_title('tSNE Reduction')

plt.show()



