#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 13:25:43 2020

@author: peymantehrani
"""

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class PolicyNetwork(nn.Module):
    def __init__(self, lr, state_dims, n_actions):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dims, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x




class PolicyGradientAgent():
    def __init__(self, lr, state_dims, gamma=0.99, n_actions=4):
        self.gamma = gamma
        self.lr = lr
        self.reward_memory = []
        self.action_memory = []

        self.policy = PolicyNetwork(self.lr, state_dims, n_actions)

    def choose_action(self, observation):
        state = T.Tensor([observation]).to(self.policy.device).squeeze()
        probabilities = F.softmax(self.policy.forward(state))
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.action_memory.append(log_probs)

        return action.detach()

    def store_rewards(self, reward):
        self.reward_memory.append(reward)
        

    def normalize_rewards(reward_list):
        
        reward_arr=np.asarray(reward_list)
        mean_reward=np.mean(reward_arr,axis=0)
        std_reward=np.std(reward_arr,axis=0)
        normalized_reward=(reward_arr-mean_reward)/(std_reward+1e-8)
        
        return normalized_reward
   
    def save_model(self,path):
        print('...save model....')
        T.save(self.policy, path)
        return

    def load_model(self,path):
        print('...load model....')
        self.policy=T.load(path)
        return
    
    def learn(self):
        self.policy.optimizer.zero_grad()
        
        normalized_reward=PolicyGradientAgent.normalize_rewards(self.reward_memory )
        
        G = np.zeros_like(normalized_reward, dtype=np.float64)
        for t in range(normalized_reward.shape[0]):
            G_sum = 0
            discount = 1
            for k in range(t, normalized_reward.shape[0]):
                G_sum += normalized_reward[k,:] * discount
                discount *= self.gamma
            G[t] = G_sum
        G = T.tensor(G, dtype=T.float).to(self.policy.device)
        loss = 0
        for g, logprob in zip(G, self.action_memory):
            loss += -g * logprob

        loss.mean().backward()
        self.policy.optimizer.step()

        self.action_memory = []
        self.reward_memory = []        