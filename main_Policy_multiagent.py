#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 14:38:19 2021

@author: peymantehrani
"""


import numpy as np
from Reinforce_Pytorch import PolicyGradientAgent
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

    load_checkpoint = False
    state_num = env.state_num
    action_num = env.power_num 
    
    num_of_agents=n_x*n_y
    Agents_list=[]
    reward_lists_of_list=[]
    mean_reward_lists_of_list=[]
    
    
    



    global_agent = PolicyGradientAgent(lr=0.0003,state_dims=state_num,gamma=0.99,n_actions=action_num)


    for n in range(num_of_agents):
        Agents_list.append(PolicyGradientAgent(lr=0.0003,state_dims=state_num,gamma=0.99,n_actions=action_num))
        reward_lists_of_list.append([])
        mean_reward_lists_of_list.append([])
    # if load_checkpoint:
    #     agent.load_models()

    # fname = 'DQN_pytorch'  + '_lr' + str(agent.lr) +'_' 
    # figure_file = agent.chkpt_dir+'/plots/' + fname + '.png'

    interval = 100
    AggPer=1000
    st = time.time()
    reward_hist = list()
    all_reward=[]
    mean_reward=[]
    a=np.zeros(maxM*num_of_agents).astype(np.int64)
    p=np.zeros(maxM*num_of_agents)
    agent_rewards=np.zeros((num_of_agents,max_episode))

    for k in range(max_episode):
        reward_dqn_list = []
        s_actor, _ = env.reset()
        for i in range(int(Ns)-1):
            for n in range(num_of_agents):
                s_actor_agent=s_actor[n*maxM:(n+1)*maxM]
                agent=Agents_list[n]
                a_agent = agent.choose_action(s_actor_agent)
                
                p_agent=env.get_power_set(min_p)[a_agent]
                
                a[n*maxM:(n+1)*maxM]=a_agent
                p[n*maxM:(n+1)*maxM]=p_agent

                
            s_actor_next, _, rate, r,rate_all_agents = env.step(p)
            
            for n in range(num_of_agents):
                agent=Agents_list[n]
                agent.store_rewards(rate[n*maxM:(n+1)*maxM])
                reward_lists_of_list[n].append(np.mean(rate_all_agents[n*maxM:(n+1)*maxM]))
                
            s_actor = s_actor_next
            reward_dqn_list.append(r)
            all_reward.append(r)
            
        for n in range(num_of_agents):
            agent=Agents_list[n]
            agent.learn()
            agent_rewards[n,k] =np.mean(reward_lists_of_list[n][-(Ns-1):])
            
            

        if k % AggPer == 0:
            global_dict = global_agent.policy.state_dict()
            for kd in global_dict.keys():
                global_dict[kd] = torch.stack([Agents_list[n].policy.state_dict()[kd] for n in range(num_of_agents)], 0).mean(0)
            global_agent.policy.load_state_dict(global_dict)
            for n in range(num_of_agents):
                Agents_list[n].policy.load_state_dict(global_agent.policy.state_dict())
            
        reward_hist.append(np.mean(reward_dqn_list))   # bps/Hz per link
        if k % interval == 0: 
            reward = np.mean(reward_hist[-interval:])
            mean_reward.append(reward)
            for n in range(num_of_agents):
                mean_reward_lists_of_list[n].append(np.mean(agent_rewards[n,-interval:]))
            if reward > max_reward:
                print("...Saving Model...")
                #agent.save_model('models/DPG_dist.pth')
                global_agent.save_model('models/DPG_Fed_AggPer_'+str(AggPer)+'.pth')

                max_reward = reward
            print("Episode(train):%d  Multi_agent Policy: %.3f  Time cost: %.2fs" %(k, reward, time.time()-st))
            st = time.time()


plt.plot(mean_reward)

np.save('npfiles/dist_Policy_nx_'+str(n_x)+'_ny_'+str(n_y)+'_m_'+str(maxM)+'.npy',np.array(mean_reward))


