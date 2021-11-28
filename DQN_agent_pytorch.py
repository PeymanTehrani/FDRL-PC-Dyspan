#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 11:21:02 2020

@author: peymantehrani
"""

import numpy as np
import torch as T
from DeepQN_Model import DeepQNetwork
from Experience_replay import ReplayBuffer

class DQNAgent(object):
    def __init__(self,gamma, lr, n_actions, state_dim,
                 buffer_size, batch_size, INITIAL_EPSILON, FINAL_EPSILON,max_episode,
                 replace=1000,chkpt_dir='tmp/dqn'):
        self.gamma = gamma
        self.epsilon = INITIAL_EPSILON
        self.lr = lr
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.batch_size = batch_size
        self.INITIAL_EPSILON = INITIAL_EPSILON
        self.FINAL_EPSILON = FINAL_EPSILON
        self.max_episode=max_episode
        self.replace_target_cnt = replace
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0
        
        #self.memory = ReplayBuffer(mem_size, input_dims, n_actions)
        self.memory =ReplayBuffer(buffer_size)

        self.q_eval = DeepQNetwork(self.lr, self.n_actions,
                                    state_dim=self.state_dim,
                                    chkpt_dir=self.chkpt_dir)

        self.q_next = DeepQNetwork(self.lr, self.n_actions,
                                    state_dim=self.state_dim,
                                    chkpt_dir=self.chkpt_dir)
        
        self.q_next.load_state_dict(self.q_eval.state_dict())
        self.tau=0.001
        
    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation],dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)

            action = T.argmax(T.squeeze(actions),dim=1).detach().numpy()
        else:
            action = np.random.choice(self.action_space)
        
        return action
    
    def select_action(self, observation,episode):
     epsilon  = self.INITIAL_EPSILON - episode * (self.INITIAL_EPSILON - self.FINAL_EPSILON) / self.max_episode
     M=observation.shape[0]
     state = T.tensor([observation],dtype=T.float).to(self.q_eval.device)
     actions = self.q_eval.forward(state)
     a_hat=T.argmax(T.squeeze(actions),dim=1).detach().numpy()
     random_index = np.array(np.random.uniform(size = (M)) < epsilon, dtype = np.int32)
     random_action = np.random.randint(0, high = self.n_actions , size = (M))
     action_set = np.vstack([a_hat, random_action])
     a = action_set[random_index, range(M)] #[M]
     #p = self.power_set[power_index] # W
     #a = np.zeros((M, self.n_actions ), dtype = np.float32)
     #a[range(self.M), power_index] = 1.
     return  a

    def store_transition(self, state, action, reward, state_):
        #self.memory.store_transition(state, action, reward, state_, done)
        self.memory.add(state, action, reward, state_)
    
    # def convert_action_to_power(self,action):
    #     return self.power_set[action]

    def sample_memory(self):
        state, action, reward, new_state = self.memory.sample_batch(self.batch_size)

        states = T.tensor(state,dtype=T.float).to(self.q_eval.device)
        rewards = T.tensor(reward,dtype=T.float).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state,dtype=T.float).to(self.q_eval.device)

        return states, actions, rewards, states_

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self,episode):
        self.epsilon = self.INITIAL_EPSILON - episode * (self.INITIAL_EPSILON - self.FINAL_EPSILON) / self.max_episode


    def save_models(self,save_file):
        self.q_eval.save_checkpoint(save_file)
        #self.q_next.save_checkpoint()

    def load_models(self,load_file):
        self.q_eval.load_checkpoint(load_file)
        #self.q_next.load_checkpoint()

    def learn(self):


        self.q_eval.optimizer.zero_grad()

        #self.replace_target_network()

        states, actions, rewards, states_ = self.sample_memory()
        batch_size=min(self.batch_size,states.shape[0])
        indices = np.arange(batch_size)


        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_).max(dim=1)[0]

        #q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next
        #q_target = rewards
        

        
            
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        for k in self.q_next.state_dict().keys():
            self.q_next.state_dict()[k]=self.tau*self.q_eval.state_dict()[k] + (1. - self.tau)*self.q_next.state_dict()[k]

