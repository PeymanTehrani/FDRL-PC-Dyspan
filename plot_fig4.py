#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 20:15:15 2020

@author: peymantehrani
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


#np.save('npfiles/FLDQN_mean_reward.npy',np.array(mean_reward))

# reward_dic = {}
# reward_dic['mean_reward'] = np.array(mean_reward)
# sio.savemat('npfiles/FLDQN_mean_reward.mat',reward_dic)


####### PLOT POLICY FEDERATED #######



# FL_poly_agg1=np.load('npfiles/FL_Policy__mean_reward_agg1.npy')
# FL_poly_agg10=np.load('npfiles/FL_Policy__mean_reward_agg10.npy')
# FL_poly_agg100=np.load('npfiles/FL_Policy__mean_reward_agg100.npy')
# FL_poly_agg1000=np.load('npfiles/FL_Policy__mean_reward_agg1000.npy')
# dist_poly=np.load('npfiles/distributed_Policy__mean_reward.npy')
# central_poly=np.load('npfiles/fully_central_Reinforce__mean_reward.npy')

# plt.figure(figsize=(12, 8))
# plt.plot(FL_poly_agg1,marker='v',linewidth=5,label="Agg Period =1",markeredgewidth=5,markerfacecolor='w')
# plt.plot(FL_poly_agg10,marker='^',linewidth=5,label="Agg Period =10",markeredgewidth=5,markerfacecolor='w')
# plt.plot(FL_poly_agg100,marker='<',linewidth=5,label="Agg Period =100",markeredgewidth=5,markerfacecolor='w')
# plt.plot(FL_poly_agg1000,marker='>',linewidth=5,label="Agg Period =1000",markeredgewidth=5,mfc='none')
# plt.plot(dist_poly,marker='8',linewidth=5,label="Distributed",markeredgewidth=5,markerfacecolor='w')
# plt.plot(central_poly,marker='o',linewidth=5,label="Centralized",markeredgewidth=5,markerfacecolor='w')

# plt.xlabel('Iteration',fontsize=22)
# plt.ylabel('Network Sum Rate',fontsize=22)
# plt.grid(b=None, which='major', axis='both')
# plt.title('Deep Policy Gradient', fontsize=22)
# plt.legend(loc="upper left",prop={'size': 16})
# plt.xticks(size = 20)
# plt.yticks(size = 20)

# plt.savefig('figs/FL_Policy_gradients.png')
# plt.savefig('figs/FL_Policy_gradients.eps', format='eps')

###########MATLAB####

# reward_dic = {}
# reward_dic['FL_poly_agg1'] = FL_poly_agg1
# sio.savemat('matfiles/FL_poly_agg1.mat',reward_dic)

# reward_dic = {}
# reward_dic['FL_poly_agg10'] = FL_poly_agg10
# sio.savemat('matfiles/FL_poly_agg10.mat',reward_dic)

# reward_dic = {}
# reward_dic['FL_poly_agg100'] = FL_poly_agg100
# sio.savemat('matfiles/FL_poly_agg100.mat',reward_dic)

# reward_dic = {}
# reward_dic['FL_poly_agg1000'] = FL_poly_agg1000
# sio.savemat('matfiles/FL_poly_agg1000.mat',reward_dic)

# reward_dic = {}
# reward_dic['dist_poly'] = dist_poly
# sio.savemat('matfiles/dist_poly.mat',reward_dic)

# reward_dic = {}
# reward_dic['central_poly'] = central_poly
# sio.savemat('matfiles/central_poly.mat',reward_dic)

####### PLOT DQN FEDERATED #######



# FL_DQN_agg1=np.load('npfiles/FL_DQN__mean_reward_agg1.npy')
# FL_DQN_agg10=np.load('npfiles/FL_DQN__mean_reward_agg10.npy')
# FL_DQN_agg100=np.load('npfiles/FL_DQN__mean_reward_agg100.npy')
# FL_DQN__agg1000=np.load('npfiles/FL_DQN__mean_reward_agg1000.npy')
# dist_DQN=np.load('npfiles/distributed_DQN__mean_reward.npy')
# cent_DQN=np.load('npfiles/fully_central_DQN__mean_reward.npy')


# plt.figure(figsize=(12, 8))
# plt.plot(FL_DQN_agg1,marker='1',linewidth=5,label="Agg Period =1")
# plt.plot(FL_DQN_agg10,marker='2',linewidth=5,label="Agg Period =10")
# plt.plot(FL_DQN_agg100,marker='3',linewidth=5,label="Agg Period =100")
# plt.plot(FL_DQN__agg1000,marker='4',linewidth=5,label="Agg Period =1000")
# plt.plot(dist_DQN,marker='*',linewidth=5,label="Distributed")
# plt.plot(cent_DQN,marker='o',linewidth=5,label="Centralized")

# plt.xlabel('Iteration',fontsize=22)
# plt.ylabel('Network Sum Rate',fontsize=22)
# plt.grid(b=None, which='major', axis='both')
# plt.title('DQN', fontsize=22)
# plt.legend(loc="upper left",prop={'size': 16})
# plt.xticks(size = 20)
# plt.yticks(size = 20)

# plt.savefig('figs/FL_DQN_central.eps', format='eps')
# plt.savefig('figs/FL_DQN_central.png')



# ###########MATLAB####

# reward_dic = {}
# reward_dic['FL_DQN_agg1'] = FL_DQN_agg1
# sio.savemat('matfiles/FL_DQN_agg1.mat',reward_dic)

# reward_dic = {}
# reward_dic['FL_DQN_agg10'] = FL_DQN_agg10
# sio.savemat('matfiles/FL_DQN_agg10.mat',reward_dic)

# reward_dic = {}
# reward_dic['FL_DQN_agg100'] = FL_DQN_agg100
# sio.savemat('matfiles/FL_DQN_agg100.mat',reward_dic)

# reward_dic = {}
# reward_dic['FL_DQN__agg1000'] = FL_DQN__agg1000
# sio.savemat('matfiles/FL_DQN__agg1000.mat',reward_dic)

# reward_dic = {}
# reward_dic['dist_DQN'] = dist_DQN
# sio.savemat('matfiles/dist_DQN.mat',reward_dic)

# reward_dic = {}
# reward_dic['cent_DQN'] = cent_DQN
# sio.savemat('matfiles/cent_DQN.mat',reward_dic)



#######################



# TD3_10=np.load('npfiles/TD3_mean_reward_policy_freq_10.npy')
# TD3_20=np.load('npfiles/TD3_mean_reward_policy_freq_20.npy')
# TD3_40=np.load('npfiles/TD3_mean_reward_policy_freq_40.npy')
# TD3_80=np.load('npfiles/TD3_mean_reward_policy_freq_80.npy')
# TD3_160=np.load('npfiles/TD3_mean_reward_policy_freq_160.npy')

# plt.figure(figsize=(15,10))
# plt.plot(TD3_10,marker='1',linewidth=5,label="Actor Update Period=10")
# plt.plot(TD3_20,marker='2',linewidth=5,label="Actor Update Period=20")
# plt.plot(TD3_40,marker='3',linewidth=5,label="Actor Update Period=40")
# plt.plot(TD3_80,marker='4',linewidth=5,label="Actor Update Period=80")
# plt.plot(TD3_160,marker='H',linewidth=5,label="Actor Update Period=160")

# plt.xlabel('Iteration',fontsize=22)
# plt.ylabel('Network Sum Rate',fontsize=22)
# plt.grid(b=None, which='major', axis='both')
# plt.title('TD3', fontsize=22)
# plt.legend(loc="best",prop={'size': 16})
# plt.xticks(size = 20)
# plt.yticks(size = 20)

# plt.savefig('figs/TD3_converge.eps', format='eps')
# plt.savefig('figs/TD3_converge.png')



#######################################
maxM=4
n_x=2
n_y=2

dist_BS_small=np.load('npfiles/dist_Policy_nx_'+str(n_x)+'_ny_'+str(n_y)+'_m_'+str(maxM)+'.npy')[-20:].mean()
FL_BS_small=np.load('npfiles/FL_Policy_nx_'+str(n_x)+'_ny_'+str(n_y)+'_m_'+str(maxM)+'.npy')[-20:].mean()

n_x=4
n_y=4

dist_BS_medium=np.load('npfiles/dist_Policy_nx_'+str(n_x)+'_ny_'+str(n_y)+'_m_'+str(maxM)+'.npy')[-20:].mean()
FL_BS_medium=np.load('npfiles/FL_Policy_nx_'+str(n_x)+'_ny_'+str(n_y)+'_m_'+str(maxM)+'.npy')[-20:].mean()

n_x=6
n_y=6

dist_BS_large=np.load('npfiles/dist_Policy_nx_'+str(n_x)+'_ny_'+str(n_y)+'_m_'+str(maxM)+'.npy')[-20:].mean()
FL_BS_large=np.load('npfiles/FL_Policy_nx_'+str(n_x)+'_ny_'+str(n_y)+'_m_'+str(maxM)+'.npy')[-20:].mean()

dist=[dist_BS_small,dist_BS_medium,dist_BS_large]
FL=[FL_BS_small,FL_BS_medium,FL_BS_large]
print((np.asarray(FL)-np.asarray(dist))/np.asarray(dist))

labels = ['BS=4', 'BS=20', 'BS=36']


x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, dist, width, label='dist')
rects2 = ax.bar(x + width/2, FL, width, label='FL')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Per User Rate')
ax.set_title('Policy gradient ')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.grid()

fig.tight_layout()
plt.savefig('figs/bar_policy_BSs.eps', format='eps')

plt.show()





##########################################





