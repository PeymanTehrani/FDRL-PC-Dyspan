#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 18:58:12 2020

@author: peymantehrani
"""

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
from Reinforce_Pytorch import PolicyGradientAgent
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

    load_checkpoint = False
    state_num = env.state_num
    action_num = env.power_num
        
    agent = PolicyGradientAgent(lr=0.0003,state_dims=state_num,gamma=0.99,n_actions=action_num)

    interval = 100
    st = time.time()
    reward_hist = list()
    all_reward=[]
    mean_reward=[]
    for k in range(max_episode):
        reward_policy_list = []
        s_actor, _ = env.reset()
        for i in range(int(Ns)-1):
            a = agent.choose_action(s_actor)
            p=env.get_power_set(min_p)[a]
            s_actor_next, _, rate, r ,_= env.step(p)

            agent.store_rewards(rate)

            s_actor = s_actor_next
            reward_policy_list.append(r)
            all_reward.append(r)
            

        agent.learn()
            
        reward_hist.append(np.mean(reward_policy_list))   # bps/Hz per link
        if k % interval == 0: 
            reward = np.mean(reward_hist[-interval:])
            mean_reward.append(reward)
            if reward > max_reward:
                print("...Saving Model...")
                agent.save_model('models/DPG_Central.pth')
                max_reward = reward
            print("Episode(train):%d  policy: %.3f  Time cost: %.2fs" %(k, reward, time.time()-st))
            st = time.time()


plt.plot(mean_reward,marker='>',linewidth=1.5)

np.save('npfiles/fully_central_Reinforce__mean_reward.npy',np.array(mean_reward))
 

