# FDRL-PC-Dyspan
Federated Deep Reinforcement Learning for the Distributed Control of NextG Wireless Networks.

This repository contains the entire code for our work "Federated Deep Reinforcement Learning for the Distributed Control of NextG Wireless Networks" and has been
 accepted for presentation in IEEE International Symposium on Dynamic Spectrum Access Networks (DySPAN) 2021.
 https://arxiv.org/abs/2112.03465
 


# Requirements
The following versions have been tested: Python 3.7.11 + Pytorch 1.7.0 But newer versions should also be fine.



## The introduction of each file


Environment and benchmarks:

`Environment_CU.py`: the mutilcell cellular wireless Environment simulator for power control.

`Benchmark_alg.py`: bench mark class which contains 4 algorithms: WWMSE, FP, random and maxpower.

`Benchmark_test.py`: testing the benchmarh performance in an environment.



Value Badesed DRL, DQN:

`DQN_agent_pytorch.py`: The DQN agent class.

`DeepQN_Model.py`: Deep Q netwrork architechure for the DQN agent class.

`Experience_replay.py`: Exprience replar buffer class for DQN agent.

`main_dqn.py`: Centrelized Deep Q Learning main file.

`main_dqn_multiagent.py`: Federated and Distributed multi agent Deep Q Learning main file.



Policy Badesed DRL, DPG:

`Reinforce_Pytorch.py`：Deep Reinforce agent and the policy netwrok architecture.

`main_reinforce.py`: Centrelized Deep Policy Gradient (Deep Reinforce) main file.

`main_Policy_multiagent.py`: Federated and Distributed multi agent Deep policy gradient  main file.



Plots and Reults:

`plot_fig4.py`: Plotting the Figure 4 of the paper.

`optmization_DRL_compare_all.py`: Compaering the performance of all methods (Table 1 of the paper).



Actor Critic Based DRL: (These were not used for the paper)

`ddpg_agent.py`：Deep Deterministic Plocy gradient (DDPG) agent class.

`TD3.py`: Twin Delayed DDPG (TD3) agent class.

`main_ddpg.py`: Main file for train the TD3 and  DDPG agents.
