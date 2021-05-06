import os
import torch
import numpy as np
import torch.optim as optim
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import matplotlib.pyplot as plt
import pickle
import time

import gym
from gym_brt.envs import QubeBeginDownEnv
from gym_brt.envs import QubeBeginUprightEnv
from random import *

hidden_size = 128
gamma = 0.99
lamda = 0.98
batch_size = 64
clip_param = 0.2
actor_lr = 1e-3
critic_lr = 1e-4
l2_rate = 0.001
render = False

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")


class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, n):
        self._n = n

    @property
    def mean(self):
        return self._M

    @mean.setter
    def mean(self, M):
        self._M = M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class ZFilter:
    

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update: self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def output_shape(self, input_space):
        return input_space.shape

class Actor(nn.Module):
    def __init__(self, num_inputs, num_outputs,init_w=3e-3):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_outputs)
        self.fc3.weight.data.uniform_(-init_w,init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mu = self.fc3(x)
        logstd = torch.zeros_like(mu)
        std = torch.exp(logstd)
        return mu, std, logstd


class Critic(nn.Module):
    def __init__(self, num_inputs,init_w=3e-3):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        v = self.fc3(x)
        return v


def get_action(mu, std):
    action = torch.normal(mu, std)
    action = action.data.numpy()
    return action


def log_density(x, mu, std, logstd):
    var = std.pow(2)
    log_density = -(x - mu).pow(2) / (2 * var) \
                  - 0.5 * math.log(2 * math.pi) - logstd
    return log_density.sum(1, keepdim=True)


def get_gae(rewards, masks, values):
    rewards = torch.Tensor(rewards)
    masks = torch.Tensor(masks)
    returns = torch.zeros_like(rewards)
    advants = torch.zeros_like(rewards)

    running_returns = 0
    previous_value = 0
    running_advants = 0

    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + gamma * running_returns * masks[t]
        running_tderror = rewards[t] + gamma * previous_value * masks[t] - \
                    values.data[t]
        running_advants = running_tderror + gamma * lamda * \
                          running_advants * masks[t]

        returns[t] = running_returns
        previous_value = values.data[t]
        advants[t] = running_advants

    advants = (advants - advants.mean()) / advants.std()
    return returns, advants


def surrogate_loss(actor, advants, states, old_policy, actions, index):
    mu, std, logstd = actor(torch.Tensor(states))
    new_policy = log_density(actions, mu, std, logstd)
    old_policy = old_policy[index]

    ratio = torch.exp(new_policy - old_policy)
    surrogate = ratio * advants
    return surrogate, ratio


def train_model(actor, critic, memory, actor_optim, critic_optim):
    memory = np.array(memory)
    states = np.vstack(memory[:, 0])
    actions = list(memory[:, 1])
    rewards = list(memory[:, 2])
    masks = list(memory[:, 3])
    values = critic(torch.Tensor(states))

    
    returns, advants = get_gae(rewards, masks, values)
    mu, std, logstd = actor(torch.Tensor(states))
    old_policy = log_density(torch.Tensor(actions), mu, std, logstd)
    old_values = critic(torch.Tensor(states))

    criterion = torch.nn.MSELoss()
    n = len(states)
    arr = np.arange(n)

    losses = torch.zeros(int(n // batch_size))

    
    for epoch in range(10):
        np.random.shuffle(arr)

        for i in range(n // batch_size):
            batch_index = arr[batch_size * i: batch_size * (i + 1)]
            batch_index = torch.LongTensor(batch_index)
            inputs = torch.Tensor(states)[batch_index]
            returns_samples = returns.unsqueeze(1)[batch_index]
            advants_samples = advants.unsqueeze(1)[batch_index]
            actions_samples = torch.Tensor(actions)[batch_index]
            oldvalue_samples = old_values[batch_index].detach()

            loss, ratio = surrogate_loss(actor, advants_samples, inputs,
                                         old_policy.detach(), actions_samples,
                                         batch_index)

            values = critic(inputs)
            clipped_values = oldvalue_samples + \
                             torch.clamp(values - oldvalue_samples,
                                         -clip_param,
                                         clip_param)
            critic_loss1 = criterion(clipped_values, returns_samples)
            critic_loss2 = criterion(values, returns_samples)
            critic_loss = torch.max(critic_loss1, critic_loss2).mean()

            clipped_ratio = torch.clamp(ratio,
                                        1.0 - clip_param,
                                        1.0 + clip_param)
            clipped_loss = clipped_ratio * advants_samples
            actor_loss = -torch.min(loss, clipped_loss).mean()

            loss = actor_loss + 0.5 * critic_loss
            losses[i] = loss
            critic_optim.zero_grad()
            loss.backward(retain_graph=True)
            critic_optim.step()

            actor_optim.zero_grad()
            loss.backward()
            actor_optim.step()
    return losses

if __name__=="__main__":
    env = QubeBeginUprightEnv()    
    env.seed(500)
    torch.manual_seed(500)
    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    print('state size:', num_inputs)
    print('action size:', num_actions)

    actor = Actor(num_inputs, num_actions).to(device)
    critic = Critic(num_inputs).to(device)

    running_state = ZFilter((num_inputs,), clip=5)

    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=critic_lr,
                              weight_decay=l2_rate)

    episodes = 0
    reward_list = []
    losses = []
    plt.ion()
    for iteration in range(1,15000):
        actor.eval(), critic.eval()
        memory = deque()

        steps = 0
        scores = []
        episode_score = 0
        while steps < 2000:
            episodes += 1
            state = env.reset()
            state = running_state(state)
            
            start_vect=time.time()
            for _ in range(2000):
                steps += 1
                mu, std, _ = actor(torch.Tensor(state).unsqueeze(0))
                action = get_action(mu, std)[0]
                print(action)
                next_state, reward, done, _ = env.step(action) 
                next_state = running_state(next_state)

                if done:
                    mask = 0
                else:
                    mask = 1

                memory.append([state, action, reward, mask])

                episode_score += reward
                state = next_state

                if done:
                    break
            print("training Runtime: %0.2f seconds"%(time.time() - start_vect))
            env.step([0.0])
        print('{} episode score is {:.2f}'.format(episodes, episode_score))
        actor.train(), critic.train()
        loss = train_model(actor, critic, memory, actor_optim, critic_optim)
        losses.append(loss.mean().detach().numpy())
        reward_list.append(episode_score)
        plt.figure(1,figsize=(20,5))
        plt.subplot(1,2,1)
        plt.title('iteration %s. reward: %s' % (iteration, reward_list[-1]))
        plt.plot(list(range(1,iteration+1)),reward_list,'b')

        plt.subplot(1,2,2)
        plt.title('iteration %s. loss: %s' % (iteration, losses[-1]))
        plt.plot(list(range(1,iteration+1)),losses,'darkorange')

        plt.pause(0.001)            
        plt.show()

        if iteration % 50 == 0:
            plt.savefig('/home/ctrllab/19_urp/code_test/qube_servo/result_img/bal_ppo_img'+str(iteration)+'.png')
            with open('/home/ctrllab/19_urp/code_test/qube_servo/result_data/bal_ppo_result_reward_'+str(iteration)+'.pickle','wb') as savedata:
                pickle.dump([iteration,reward_list],savedata)
            with open('/home/ctrllab/19_urp/code_test/qube_servo/result_data/bal_ppo_result_loss_'+str(iteration)+'.pickle','wb') as savedata:
                pickle.dump([iteration,losses],savedata)
            torch.save(actor.state_dict(),'./ppo_real_'+str(iteration)+'.pth')
    env.close()    
    plt.waitforbuttonpress(0)
    plt.close()
