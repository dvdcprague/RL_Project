#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 14:43:47 2017

@author: cc
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 14:24:05 2017

@author: cc
"""

import gym
from gym import wrappers
import numpy as np
import random
import DQNLearner as dqn
from datalogger import ExperimentLogger

GAME = 'LunarLander-v2'
env = gym.make(GAME)
env.seed(16)
np.random.seed(16)
random.seed(16)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

RECORD = None
MAX_EPISODES = 2000
MAX_EPISODES_TEST = 200

logger_train = ExperimentLogger(logdir="./result", prefix=("result_train"), num_episodes=MAX_EPISODES, verbose=True)

env = wrappers.Monitor(env, 'upload/'+GAME, force=True, video_callable=RECORD)

agent = dqn.DQNLearner(state_dim=state_dim,\
        action_dim = action_dim, \
        alpha = 0.0005, \
        gamma = 0.99, \
        rar = 1.0, \
        radr = 0.975, \
        units = 40, \
        update_target_freq=600, \
        verbose = False) #initialize the learner
                                                  
for episode in range(MAX_EPISODES):
    
    total_reward = 0
    loss = 0
    step = 0
    state = env.reset()
    
    while True:
        action = agent.epsilon_greedy_action(state)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        agent.store_experience(state, action, reward, next_state, done)
        loss += agent.train()
        state = next_state
        step += 1
        if done :
            break
#    print("episode: ", episode, "total reward: ", total_reward)
    logger_train.log_episode(total_reward, loss, episode)
#    
#    if episode == MAX_EPISODES-1:
#        print("Test Period")
#        for i in range(MAX_EPISODES_TEST):
#            total_reward_test = 0
#            loss_test = 0
#            step_test = 0
#            state_test = env.reset()
#            while True:
#                env.render()
#                action_test = agent.set_action(state_test)
#                next_state_test,reward_test,done_test,info_test = env.step(action_test)
#                total_reward_test += reward_test
#                state_test = next_state_test
#                step_test += 1
#                if done_test:
#                    break
#            print("episode: ", i, "total reward: ", total_reward_test)

env.close()    
gym.upload('upload/'+GAME, api_key='sk_ocA2j8g2QyqixgtrtVbOSA')
