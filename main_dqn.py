#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 20:13:15 2020

@author: peymantehrani
"""

import numpy as np
from DQN_agent_pytorch import DQNAgent
from Environment_CU import Env_cellular
import time
import matplotlib.pyplot as plt
import torch

if __name__ == '__main__':
    fd = 10
    Ts = 20e-3
    n_x = 5
    n_y = 5
    L = 2
    C = 16
    maxM = 4   # user number in one BS
    min_dis = 0.01 #km
    max_dis = 1. #km
    max_p = 38. #dBm
    min_p = 5
    p_n = -114. #dBm
    power_num = 10  #action_num
    seed=11
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = Env_cellular(fd, Ts, n_x, n_y, L, C, maxM, min_dis, max_dis, max_p, p_n, power_num)
    
    max_reward = 0
    batch_size = 500
    max_episode = 7000
    buffer_size = 50000
    Ns = 11
    env.set_Ns(Ns) 
    INITIAL_EPSILON = 0.2 
    FINAL_EPSILON = 0.0001
    load_checkpoint = False
    state_num = env.state_num
    action_num = env.power_num


    agent = DQNAgent( gamma=0, lr=1e-3, n_actions=action_num, state_dim=state_num,
                 buffer_size=buffer_size,batch_size=batch_size, INITIAL_EPSILON= 0.2, FINAL_EPSILON=0.0001,max_episode=7000,
                 replace=1000)

        

    # if load_checkpoint:
    #     agent.load_models()

    # fname = 'DQN_pytorch'  + '_lr' + str(agent.lr) +'_' 
    # figure_file = agent.chkpt_dir+'/plots/' + fname + '.png'

    interval = 100
    st = time.time()
    reward_hist = list()
    all_reward=[]
    mean_reward=[]
    for k in range(max_episode):
        reward_dqn_list = []
        s_actor, _ = env.reset()
        for i in range(int(Ns)-1):
            a = agent.select_action(s_actor,k)
            p=env.get_power_set(min_p)[a]
            s_actor_next, _, rate, r , _= env.step(p)

            agent.store_transition(s_actor, a, rate,s_actor_next)
            s_actor = s_actor_next
            reward_dqn_list.append(r)
            all_reward.append(r)
            

        agent.learn()
            
        reward_hist.append(np.mean(reward_dqn_list))   # bps/Hz per link
        if k % interval == 0: 
            reward = np.mean(reward_hist[-interval:])
            mean_reward.append(reward)
            if reward > max_reward:
                agent.save_models('models/DQN_central.pth')
                max_reward = reward       

            print("Episode(train):%d  DQN: %.3f  Time cost: %.2fs" %(k, reward, time.time()-st))
            st = time.time()


plt.plot(mean_reward)
 
np.save('npfiles/fully_central_DQN__mean_reward.npy',np.array(mean_reward))