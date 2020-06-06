from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.autograd as autograd
from PIL import Image
import math, random
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
USE_CUDA = torch.cuda.is_available()

class QLearner(nn.Module):
    def __init__(self, env, num_frames, batch_size, gamma, replay_buffer):
        super(QLearner, self).__init__()

        self.batch_size = batch_size
        self.gamma = gamma
        self.num_frames = num_frames
        self.replay_buffer = replay_buffer
        self.env = env
        self.input_shape = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n

        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def feature_size(self):
            return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            #Image.fromarray(state[0,:,:], 'L').save('states/t.png')
            state   = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), requires_grad=True)
            q_value = self.forward(state)
            action = torch.argmax(q_value).data.cpu()

        else:
            action = random.randrange(self.env.action_space.n)

        return action
        
def compute_td_loss(model, target_model, batch_size, gamma, replay_buffer):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)), requires_grad=True)
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    #done = Variable(torch.FloatTensor(done))

    qnext = target_model.forward(next_state)
    qnext_max = qnext.max(axis=1)[0]

    qthis = model.forward(state)
    qthis_max = qthis.gather(1, action.unsqueeze(1)).view(-1) # must use action, since sometimes action is non-optimal

    target = reward + (gamma * qnext_max)
    target = target.view(-1)

    batch_loss = (target.detach() - qthis_max).pow(2)

    loss = batch_loss.sum()

    return loss



class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(list(self.buffer), batch_size)

        state = [tup[0] for tup in batch]
        action = [tup[1] for tup in batch]
        reward = [tup[2] for tup in batch]
        next_state = [tup[3] for tup in batch]
        done = [tup[4] for tup in batch]

        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)
