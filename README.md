# RL-Control-Furuta-Pendulum

codes for control of furuta pendulum with reinforcement learning.

swing up for ddpg, balancing for ppo

experiment video link(youtube): https://www.youtube.com/watch?v=a6W6u8iMDU8

ddpg hyperparameter | value  
------------ | ------------  
state space dim | 10  
control space dim | 1  
total num. policy params | 40960  
steps per iteration | 3000  
training iteration | 100
minibatch size | 64
discounted factor(gamma) | 0.99
critic learning rate | 1e-3
actor learning rate | 1e-4
soft target update parameter(tau) | 0.01



ppo hyperparameter | value  
------------ | ------------  
state space dim | 10  
control space dim | 1  
total num. policy params | 163840  
steps per iteration | 2000  
training iteration | 100
minibatch size | 64
discounted factor(gamma) | 0.99
critic learning rate | 1e-4
actor learning rate | 1e-3
GAE parameter(lambda) | 0.98
clipping parameter(elipsion) | 0.2
VF coefficient(c_1) | 0.5
Entropy coefficient(c_2) | 0


reference

[1] https://github.com/higgsfield/RL-Adventure-2/blob/master/5.ddpg.ipynb

[2] https://github.com/reinforcement-learning-kr/pg_travel/blob/master/mujoco/agent/ppo_gae.py
