#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 12:34:12 2021

@author: peymantehrani
"""
import numpy as np
from Environment_CU import Env_cellular
from Benchmark_alg import Benchmark_alg
from Reinforce_Pytorch import PolicyGradientAgent
from DQN_agent_pytorch import DQNAgent
from Benchmark_test import Benchmark_test_time
import time
import torch
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
power_num = 10 #action_num
seed=11
Ns=21
max_episode=100 #for test
#max_episode=100 #for test
env = Env_cellular(fd, Ts, n_x, n_y, L, C, maxM, min_dis, max_dis, max_p, p_n, power_num)
env.set_Ns(Ns)

batch_size = 500
buffer_size = 50000
INITIAL_EPSILON = 0.2 
FINAL_EPSILON = 0.0001
state_num = env.state_num
action_num = env.power_num  
num_of_agents=n_x*n_y
a=np.zeros(n_x*n_y*maxM).astype(np.int64)
p=np.zeros(n_x*n_y*maxM)
dqn_agent=DQNAgent( gamma=0, lr=1e-3, n_actions=action_num, state_dim=state_num,
                 buffer_size=buffer_size,batch_size=batch_size, INITIAL_EPSILON= 0, FINAL_EPSILON=0,max_episode=5000,
                 replace=1000)
DPG_agent=PolicyGradientAgent(lr=0.0003,state_dims=state_num,gamma=0.99,n_actions=action_num)
l=max_episode*(Ns-1)



bench = Benchmark_alg(env)
reward_hist = list()
for k in range(1, max_episode+1):
    reward_list = list()
    s_actor,s = env.reset_()
    for i in range(int(Ns)-1):
        p = bench.calculate(s)
        s_next, r = env.step__(p)
        s = s_next
        reward_list.append(r)
    reward_hist.append(reward_list)
    
reward_hist = np.reshape(reward_hist, [max_episode, Ns-1, 4])
reward_mean = np.nanmean(np.nanmean(reward_hist, 0), 0)
reward_std = np.nanstd(np.nanmean(reward_hist, 0), 0)
time_costs_benchmarks=Benchmark_test_time(max_episode, Ns, fd, max_dis, maxM)
print("Bench Marks Mean Rate: [FP, WMSSE, MAX_POWER, Random_POWER] : ",reward_mean)
print("Bench Marks Std Rate: [FP, WMSSE, MAX_POWER, Random_POWER] : ",reward_std)
print("Bench Marks executaion_time_cost: [FP, WMSSE, MAX_POWER, Random_POWER] : ",time_costs_benchmarks)


executaion_time_cost=0
t=time.time()
torch.manual_seed(seed)
np.random.seed(seed)
dqn_agent.load_models('models/DQN_dist.pth')
agent=dqn_agent
reward_hist_dqn = list()                
for k in range(max_episode):
    s_actor, _ = env.reset()
    for i in range(int(Ns)-1):
        for n in range(num_of_agents):
            s_actor_agent=s_actor[n*maxM:(n+1)*maxM]
            st=time.time()
            a_agent = agent.select_action(s_actor_agent,k)
            executaion_time_cost=executaion_time_cost+time.time()-st
            p_agent=env.get_power_set(min_p)[a_agent]       
            a[n*maxM:(n+1)*maxM]=a_agent
            p[n*maxM:(n+1)*maxM]=p_agent
        s_actor_next, _, rate, r,rate_all_agents = env.step(p)
        s_actor = s_actor_next
        reward_hist_dqn.append(r)  
reward_hist_dqn = np.reshape(reward_hist_dqn, [max_episode*( Ns-1)])
DQN_dist_mean=np.nanmean(reward_hist_dqn)
DQN_dist_std=np.nanstd(reward_hist_dqn)
print("DQN_distributed : Mean = %.4f , STD = %.4f "%(DQN_dist_mean,DQN_dist_std))
print("elapsed time : %2f"%(time.time()-t))
print("executaion_time_cost :",executaion_time_cost/(l*num_of_agents))


executaion_time_cost=0
t=time.time()
torch.manual_seed(seed)
np.random.seed(seed)
dqn_agent.load_models('models/DQN_Fed_AggPer_1.pth')
agent=dqn_agent
reward_hist_dqn = list()                
for k in range(max_episode):
    s_actor, _ = env.reset()
    for i in range(int(Ns)-1):
        for n in range(num_of_agents):
            s_actor_agent=s_actor[n*maxM:(n+1)*maxM]
            st=time.time()
            a_agent = agent.select_action(s_actor_agent,k)
            executaion_time_cost=executaion_time_cost+time.time()-st       
            p_agent=env.get_power_set(min_p)[a_agent]       
            a[n*maxM:(n+1)*maxM]=a_agent
            p[n*maxM:(n+1)*maxM]=p_agent
        s_actor_next, _, rate, r,rate_all_agents = env.step(p)
        s_actor = s_actor_next
        reward_hist_dqn.append(r)  
reward_hist_dqn = np.reshape(reward_hist_dqn, [max_episode*( Ns-1)])
DQN_dist_mean=np.nanmean(reward_hist_dqn)
DQN_dist_std=np.nanstd(reward_hist_dqn)
print("DQN_Fed_AggPer_1 : Mean = %.4f , STD = %.4f "%(DQN_dist_mean,DQN_dist_std))
print("elapsed time : %2f"%(time.time()-t))
print("executaion_time_cost :",executaion_time_cost/(l*num_of_agents))


executaion_time_cost=0
t=time.time()
torch.manual_seed(seed)
np.random.seed(seed)
dqn_agent.load_models('models/DQN_Fed_AggPer_10.pth')
agent=dqn_agent
reward_hist_dqn = list()                
for k in range(max_episode):
    s_actor, _ = env.reset()
    for i in range(int(Ns)-1):
        for n in range(num_of_agents):
            s_actor_agent=s_actor[n*maxM:(n+1)*maxM]
            st=time.time()
            a_agent = agent.select_action(s_actor_agent,k)
            executaion_time_cost=executaion_time_cost+time.time()-st        
            p_agent=env.get_power_set(min_p)[a_agent]       
            a[n*maxM:(n+1)*maxM]=a_agent
            p[n*maxM:(n+1)*maxM]=p_agent
        s_actor_next, _, rate, r,rate_all_agents = env.step(p)
        s_actor = s_actor_next
        reward_hist_dqn.append(r)  
reward_hist_dqn = np.reshape(reward_hist_dqn, [max_episode*( Ns-1)])
DQN_dist_mean=np.nanmean(reward_hist_dqn)
DQN_dist_std=np.nanstd(reward_hist_dqn)
print("DQN_Fed_AggPer_10 : Mean = %.4f , STD = %.4f "%(DQN_dist_mean,DQN_dist_std))
print("elapsed time : %2f"%(time.time()-t))
print("executaion_time_cost :",executaion_time_cost/(l*num_of_agents))


executaion_time_cost=0
t=time.time()
torch.manual_seed(seed)
np.random.seed(seed)
dqn_agent.load_models('models/DQN_Fed_AggPer_100.pth')
agent=dqn_agent
reward_hist_dqn = list()                
for k in range(max_episode):
    s_actor, _ = env.reset()
    for i in range(int(Ns)-1):
        for n in range(num_of_agents):
            s_actor_agent=s_actor[n*maxM:(n+1)*maxM]
            st=time.time()
            a_agent = agent.select_action(s_actor_agent,k)
            executaion_time_cost=executaion_time_cost+time.time()-st  
            p_agent=env.get_power_set(min_p)[a_agent]       
            a[n*maxM:(n+1)*maxM]=a_agent
            p[n*maxM:(n+1)*maxM]=p_agent
        s_actor_next, _, rate, r,rate_all_agents = env.step(p)
        s_actor = s_actor_next
        reward_hist_dqn.append(r)  
reward_hist_dqn = np.reshape(reward_hist_dqn, [max_episode*( Ns-1)])
DQN_dist_mean=np.nanmean(reward_hist_dqn)
DQN_dist_std=np.nanstd(reward_hist_dqn)
print("DQN_Fed_AggPer_100 : Mean = %.4f , STD = %.4f "%(DQN_dist_mean,DQN_dist_std))
print("elapsed time : %2f"%(time.time()-t))
print("executaion_time_cost :",executaion_time_cost/(l*num_of_agents))


executaion_time_cost=0
t=time.time()
torch.manual_seed(seed)
np.random.seed(seed)
dqn_agent.load_models('models/DQN_Fed_AggPer_1000.pth')
agent=dqn_agent
reward_hist_dqn = list()                
for k in range(max_episode):
    s_actor, _ = env.reset()
    for i in range(int(Ns)-1):
        for n in range(num_of_agents):
            s_actor_agent=s_actor[n*maxM:(n+1)*maxM]
            st=time.time()
            a_agent = agent.select_action(s_actor_agent,k)
            executaion_time_cost=executaion_time_cost+time.time()-st  
            p_agent=env.get_power_set(min_p)[a_agent]       
            a[n*maxM:(n+1)*maxM]=a_agent
            p[n*maxM:(n+1)*maxM]=p_agent
        s_actor_next, _, rate, r,rate_all_agents = env.step(p)
        s_actor = s_actor_next
        reward_hist_dqn.append(r)  
reward_hist_dqn = np.reshape(reward_hist_dqn, [max_episode*( Ns-1)])
DQN_dist_mean=np.nanmean(reward_hist_dqn)
DQN_dist_std=np.nanstd(reward_hist_dqn)
print("DQN_Fed_AggPer_1000 : Mean = %.4f , STD = %.4f "%(DQN_dist_mean,DQN_dist_std))
print("elapsed time : %2f"%(time.time()-t))
print("executaion_time_cost :",executaion_time_cost/(l*num_of_agents))


executaion_time_cost=0
t=time.time()
torch.manual_seed(seed)
np.random.seed(seed)
dqn_agent.load_models('models/DQN_central.pth')
agent=dqn_agent
reward_hist_dqn = list()   
for k in range(max_episode):
    reward_dqn_list = []
    s_actor, _ = env.reset()
    for i in range(int(Ns)-1):
        st=time.time()
        a = agent.select_action(s_actor,k)
        executaion_time_cost=executaion_time_cost+time.time()-st
        p=env.get_power_set(min_p)[a]
        s_actor_next, _, rate, r , _= env.step(p)
        s_actor = s_actor_next
        reward_hist_dqn.append(r)
reward_hist_dqn = np.reshape(reward_hist_dqn, [max_episode*( Ns-1)])
DQN_dist_mean=np.nanmean(reward_hist_dqn)
DQN_dist_std=np.nanstd(reward_hist_dqn)
print("DQN_Central : Mean = %.4f , STD = %.4f "%(DQN_dist_mean,DQN_dist_std))
print("elapsed time : %2f"%(time.time()-t))
print("executaion_time_cost :",executaion_time_cost/(l))



executaion_time_cost=0
t=time.time()
torch.manual_seed(seed)
np.random.seed(seed)
DPG_agent.load_model('models/DPG_dist.pth')
agent=DPG_agent
reward_dpg_list = list() 
for k in range(max_episode):
    s_actor, _ = env.reset()
    for i in range(int(Ns)-1):
        for n in range(num_of_agents):
            s_actor_agent=s_actor[n*maxM:(n+1)*maxM]
            st=time.time()
            a_agent = agent.choose_action(s_actor_agent) 
            executaion_time_cost=executaion_time_cost+time.time()-st
            p_agent=env.get_power_set(min_p)[a_agent] 
            a[n*maxM:(n+1)*maxM]=a_agent
            p[n*maxM:(n+1)*maxM]=p_agent    
        s_actor_next, _, rate, r,rate_all_agents = env.step(p)
        s_actor = s_actor_next
        reward_dpg_list.append(r)  
reward_dpg_list = np.reshape(reward_dpg_list, [max_episode*( Ns-1)])
DPG_mean=np.nanmean(reward_dpg_list)
DPG_std=np.nanstd(reward_dpg_list)
print("DPG_dist : Mean = %.4f , STD = %.4f "%(DPG_mean,DPG_std))
print("elapsed time : %2f"%(time.time()-t))
print("executaion_time_cost :",executaion_time_cost/(l*num_of_agents))



executaion_time_cost=0
t=time.time()
torch.manual_seed(seed)
np.random.seed(seed)
DPG_agent.load_model('models/DPG_Fed_AggPer_1.pth')
agent=DPG_agent
reward_dpg_list = list() 
for k in range(max_episode):
    s_actor, _ = env.reset()
    for i in range(int(Ns)-1):
        for n in range(num_of_agents):
            s_actor_agent=s_actor[n*maxM:(n+1)*maxM]
            st=time.time()
            a_agent = agent.choose_action(s_actor_agent) 
            executaion_time_cost=executaion_time_cost+time.time()-st    
            p_agent=env.get_power_set(min_p)[a_agent] 
            a[n*maxM:(n+1)*maxM]=a_agent
            p[n*maxM:(n+1)*maxM]=p_agent    
        s_actor_next, _, rate, r,rate_all_agents = env.step(p)
        s_actor = s_actor_next
        reward_dpg_list.append(r)  
reward_dpg_list = np.reshape(reward_dpg_list, [max_episode*( Ns-1)])
DPG_mean=np.nanmean(reward_dpg_list)
DPG_std=np.nanstd(reward_dpg_list)
print("DPG_Fed_AggPer_1 : Mean = %.4f , STD = %.4f "%(DPG_mean,DPG_std))
print("elapsed time : %2f"%(time.time()-t))
print("executaion_time_cost :",executaion_time_cost/(l*num_of_agents))


executaion_time_cost=0
t=time.time()
torch.manual_seed(seed)
np.random.seed(seed)
DPG_agent.load_model('models/DPG_Fed_AggPer_10.pth')
agent=DPG_agent
reward_dpg_list = list() 
for k in range(max_episode):
    s_actor, _ = env.reset()
    for i in range(int(Ns)-1):
        for n in range(num_of_agents):
            s_actor_agent=s_actor[n*maxM:(n+1)*maxM]
            st=time.time()
            a_agent = agent.choose_action(s_actor_agent) 
            executaion_time_cost=executaion_time_cost+time.time()-st    
            p_agent=env.get_power_set(min_p)[a_agent] 
            a[n*maxM:(n+1)*maxM]=a_agent
            p[n*maxM:(n+1)*maxM]=p_agent    
        s_actor_next, _, rate, r,rate_all_agents = env.step(p)
        s_actor = s_actor_next
        reward_dpg_list.append(r)  
reward_dpg_list = np.reshape(reward_dpg_list, [max_episode*( Ns-1)])
DPG_mean=np.nanmean(reward_dpg_list)
DPG_std=np.nanstd(reward_dpg_list)
print("DPG_Fed_AggPer_10 : Mean = %.4f , STD = %.4f "%(DPG_mean,DPG_std))
print("elapsed time : %2f"%(time.time()-t))
print("executaion_time_cost :",executaion_time_cost/(l*num_of_agents))




executaion_time_cost=0
t=time.time()
torch.manual_seed(seed)
np.random.seed(seed)
DPG_agent.load_model('models/DPG_Fed_AggPer_100.pth')
agent=DPG_agent
reward_dpg_list = list() 
for k in range(max_episode):
    s_actor, _ = env.reset()
    for i in range(int(Ns)-1):
        for n in range(num_of_agents):
            s_actor_agent=s_actor[n*maxM:(n+1)*maxM]
            st=time.time()
            a_agent = agent.choose_action(s_actor_agent) 
            executaion_time_cost=executaion_time_cost+time.time()-st 
            p_agent=env.get_power_set(min_p)[a_agent] 
            a[n*maxM:(n+1)*maxM]=a_agent
            p[n*maxM:(n+1)*maxM]=p_agent    
        s_actor_next, _, rate, r,rate_all_agents = env.step(p)
        s_actor = s_actor_next
        reward_dpg_list.append(r)  
reward_dpg_list = np.reshape(reward_dpg_list, [max_episode*( Ns-1)])
DPG_mean=np.nanmean(reward_dpg_list)
DPG_std=np.nanstd(reward_dpg_list)
print("DPG_Fed_AggPer_100 : Mean = %.4f , STD = %.4f "%(DPG_mean,DPG_std))
print("elapsed time : %2f"%(time.time()-t))
print("executaion_time_cost :",executaion_time_cost/(l*num_of_agents))


executaion_time_cost=0
t=time.time()
torch.manual_seed(seed)
np.random.seed(seed)
DPG_agent.load_model('models/DPG_Fed_AggPer_1000.pth')
agent=DPG_agent
reward_dpg_list = list() 
for k in range(max_episode):
    s_actor, _ = env.reset()
    for i in range(int(Ns)-1):
        for n in range(num_of_agents):
            s_actor_agent=s_actor[n*maxM:(n+1)*maxM]
            st=time.time()
            a_agent = agent.choose_action(s_actor_agent) 
            executaion_time_cost=executaion_time_cost+time.time()-st      
            p_agent=env.get_power_set(min_p)[a_agent] 
            a[n*maxM:(n+1)*maxM]=a_agent
            p[n*maxM:(n+1)*maxM]=p_agent    
        s_actor_next, _, rate, r,rate_all_agents = env.step(p)
        s_actor = s_actor_next
        reward_dpg_list.append(r)  
reward_dpg_list = np.reshape(reward_dpg_list, [max_episode*( Ns-1)])
DPG_mean=np.nanmean(reward_dpg_list)
DPG_std=np.nanstd(reward_dpg_list)
print("DPG_Fed_AggPer_1000 : Mean = %.4f , STD = %.4f "%(DPG_mean,DPG_std))
print("elapsed time : %2f"%(time.time()-t))
print("executaion_time_cost :",executaion_time_cost/(l*num_of_agents))

executaion_time_cost=0
t=time.time()
torch.manual_seed(seed)
np.random.seed(seed)
DPG_agent.load_model('models/DPG_Central.pth')
agent=DPG_agent
reward_dpg_list = list() 
for k in range(max_episode):
    s_actor, _ = env.reset()
    for i in range(int(Ns)-1):
        st=time.time()
        a = agent.choose_action(s_actor)
        executaion_time_cost=executaion_time_cost+time.time()-st      
        p=env.get_power_set(min_p)[a]
        s_actor_next, _, rate, r ,_= env.step(p)
        s_actor = s_actor_next
        reward_dpg_list.append(r)
reward_dpg_list = np.reshape(reward_dpg_list, [max_episode*( Ns-1)])
DPG_mean=np.nanmean(reward_dpg_list)
DPG_std=np.nanstd(reward_dpg_list)
print("DPG_Central : Mean = %.4f , STD = %.4f "%(DPG_mean,DPG_std))
print("elapsed time : %2f"%(time.time()-t))
print("executaion_time_cost :",executaion_time_cost/(l))
