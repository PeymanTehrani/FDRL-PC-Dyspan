#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 13:55:21 2021

@author: peymantehrani
"""

import numpy as np
from ddpg_agent import Agent
from Environment_CU import Env_cellular
import time
import matplotlib.pyplot as plt
from TD3 import TD3
import torch
import scipy.io as sio


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
    max_p_watt=1e-3*pow(10,3.8)
    min_p = 5
    p_n = -114. #dBm
    power_num = 10  #action_num
    env = Env_cellular(fd, Ts, n_x, n_y, L, C, maxM, min_dis, max_dis, max_p, p_n, power_num)
    
    max_reward = 0
    max_episode = 15000
    Ns = 11
    env.set_Ns(Ns) 
    seed=11
    #state_num = env.state_num
    state_num = 50
    action_num = env.power_num
        
    #agent = Agent(alpha=0.0001*.5, beta=0.001*0.3, input_dims=state_num, tau=0.001,batch_size=128, fc1_dims=128, fc2_dims=64,n_actions=1,gamma=.0)
    torch.manual_seed(seed)
    np.random.seed(seed)
    agent=TD3(state_dim=state_num,action_dim=1,max_action=1,discount=0,policy_noise=0.005,noise_clip=0.01,policy_freq=160)
	

    interval = 100
    st = time.time()
    reward_hist = list()
    all_reward=[]
    mean_reward=[]
    critic_loss_list=[]
    actor_loss_list=[]
    target_list=[]
    critic_list=[]
    

	
    for k in range(max_episode):
        reward_policy_list = []
        s_actor, s_critic = env.reset()
        for i in range(int(Ns)-1):

            a = agent.choose_action(s_actor)
            #p=env.get_power_set(min_p)[a]
            #p=np.squeeze((a*((max_p-min_p)/2))+(max_p+min_p)/2)
            p=np.squeeze(a*(max_p_watt/2))+(max_p_watt/2)
            # p=np.squeeze(a)+(33/2+5)
            # p=1e-3*pow(10,p/10)
            s_actor_next, s_critic_next, rate, r ,_= env.step(p)
            agent.store_transition(s_actor, a, rate, s_actor_next)
        
            s_actor = s_actor_next
            s_critic=s_critic_next
            reward_policy_list.append(r)
            all_reward.append(r)

        agent.learn()
        #critic_loss,actor_loss,target,criitc_value=agent.learn()
        # critic_loss_list.append(critic_loss)
        # actor_loss_list.append(actor_loss)
        # target_list.append(target)
        # critic_list.append(criitc_value)
            
        reward_hist.append(np.mean(reward_policy_list))   # bps/Hz per link
        if k % interval == 0: 
            reward = np.mean(reward_hist[-interval:])
            mean_reward.append(reward)
            if reward > max_reward:
                #agent.save_models()
                max_reward = reward
            print("Episode(train):%d  policy: %.3f  Time cost: %.2fs" %(k, reward, time.time()-st))
            #print("critic_loss: %.3f  actor_loss: %.3f" %(critic_loss, actor_loss))
            #print("###################################################################")


            #print('POWERR==',p)
            st = time.time()


plt.figure(figsize=(9, 6))
plt.plot(mean_reward,marker='o',linewidth=3)
plt.xlabel('Iteration',fontsize=16)
plt.ylabel('Mean Reward',fontsize=16)
plt.grid(b=None, which='major', axis='both')
plt.title('TD3 Mean Reward(policy_freq_update=160)', fontsize=20)
# plt.savefig('figs/TD3_mean_reward_policy_freq_160.png')

# reward_dic = {}
# reward_dic['mean_reward'] = np.array(mean_reward)


# sio.savemat('npfiles/TD3_mean_reward_policy_freq_160.mat',reward_dic)

# plt.figure(figsize=(9, 6))
# plt.plot(critic_loss_list)
# plt.xlabel('Iteration',fontsize=16)
# plt.ylabel('Critic Loss',fontsize=16)
# plt.title('DDPG Critic Loss(with previous action)', fontsize=20)
# plt.savefig('figs/ddpg_critic_loss2.png')

# plt.figure(figsize=(9, 6))
# plt.plot(actor_loss_list)
# plt.xlabel('Iteration',fontsize=16)
# plt.ylabel('Actor Loss',fontsize=16)
# plt.title('DDPG Actor Loss (with previous action)', fontsize=20)
# plt.savefig('figs/ddpg_actor_loss2.png')














