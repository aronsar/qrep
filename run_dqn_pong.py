from Wrapper.layers import *
from Wrapper.wrappers import make_atari, wrap_deepmind, wrap_pytorch
import math, random
import gym
import numpy as np
import time
import pickle
from os import path
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
USE_CUDA = torch.cuda.is_available()
from dqn import QLearner, compute_td_loss, ReplayBuffer
import argparse 

parser = argparse.ArgumentParser() 
parser.add_argument(
    '--num_frames',
    '--nf',
    type=int, 
    default=1000000, 
    help="Number of frames to train model.") 
parser.add_argument(
    '--learning_rate',
    '--lr',
    type=float, 
    default=.00001, 
    help="Learning rate for adam optimizer.") 
parser.add_argument(
    '--epsilon_decay',
    '--ed',
    type=int, 
    default=30000, 
    help="Decay constant of the exploration probability.") 
parser.add_argument(
    '--gamma',
    '--g',
    type=float, 
    default=.99, 
    help="Discounting factor of future rewards.") 
parser.add_argument(
    '--replay_buffer',
    '--rb',
    type=int, 
    default=100000, 
    help="The size of the replay buffer.") 
parser.add_argument(
    '--copy_frequency',
    '--cf',
    type=int, 
    default=50000, 
    help="How often to copy the target model to the working model, measured in number of frames.") 
parser.add_argument(
    '--loadpath',
    '--lp',
    type=str, 
    help="Path from which to load pretrained model.") 
args = parser.parse_args()


# Set hyperparameters
num_frames = args.num_frames
batch_size = 32
gamma = args.gamma
record_idx = 10000
replay_initial = 10000
replay_buffer_size = args.replay_buffer
copy_frequency = args.copy_frequency
epsilon_start = 0.1 # Epsilon is the chance of making a random move
epsilon_final = 0.01 # It trades of exploration vs exploitation
epsilon_decay = args.epsilon_decay
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) \
    * math.exp(-1. * frame_idx / epsilon_decay)


# Initialize environment
env_id = "PongNoFrameskip-v4"
env = make_atari(env_id)
env = wrap_deepmind(env)
env = wrap_pytorch(env)
state = env.reset()


# Model initialization
replay_buffer = ReplayBuffer(replay_buffer_size)
model        = QLearner(env, num_frames, batch_size, gamma, replay_buffer)
target_model = QLearner(env, num_frames, batch_size, gamma, replay_buffer)


# Model loading
if args.loadpath and path.exists(args.loadpath):
    print("Loading model from " + args.loadpath)
    model.load_state_dict(torch.load(args.loadpath, map_location='cpu'))
else:
    print("Training new model...")
target_model.copy_from(model)


# Setting up optimizer and gpu
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
if USE_CUDA:
    model = model.cuda()
    target_model = target_model.cuda()
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
t = time.time()
logs = defaultdict(dict)


# Training
for frame_idx in range(1, num_frames + 1):
    epsilon = epsilon_by_frame(frame_idx)

    # Perform an action based on current state and epsilon
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

    
    if len(replay_buffer) > replay_initial: # replay buffer is big enough
        loss = compute_td_loss(model, target_model, batch_size, gamma, replay_buffer)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append((frame_idx, loss.data.cpu().numpy()))

    if frame_idx % 10000 == 0 and len(replay_buffer) <= replay_initial:
        print('Frame: %d, preparing replay buffer' % frame_idx)
        print('Time = %.2f' % (time.time() - t))

    if frame_idx % 10000 == 0 and len(replay_buffer) > replay_initial:
        print('Frame: %d, AvgLoss: %f' % (frame_idx, np.mean(losses, 0)[1]))
        print('Last-10k loss: %f' % np.mean(losses[-10000:], 0)[1])
        print('Last-10 average reward: %f' % np.mean(all_rewards[-10:], 0)[1])
        print('Time = %.2f' % (time.time() - t))
        logs[frame_idx]["avg_loss"] = np.mean(losses, 0)[1]
        logs[frame_idx]["last10000_loss"] = np.mean(losses[-10000:], 0)[1]
        logs[frame_idx]["last10_reward"]  = np.mean(all_rewards[-10:], 0)[1]

    # copy target model over to model (the infrequent copy)
    if frame_idx % copy_frequency == 0:
        target_model.copy_from(model)
    
    # checkpointing and logging
    if frame_idx % 100000 == 0:
        chkpt_savepath = generate_filename("./checkpoints", frame_idx, args, ".pth")
        torch.save(model.state_dict(), chkpt_savepath)

        log_savepath = generate_filename("./logs", frame_idx, args, ".pkl")
        with open(log_savepath, "wb") as f:
            pickle.dump(logs, f)
    
