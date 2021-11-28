#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 12:23:48 2020

@author: peymantehrani
"""

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, state_dim, chkpt_dir):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = self.checkpoint_dir+'/models'

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = F.relu(self.fc3(x))

        return actions

    def save_checkpoint(self,save_file):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), save_file)

    def load_checkpoint(self,load_file):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(load_file))
        
