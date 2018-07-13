#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 10:15:45 2017

@author: cc
"""

import pandas as pd
import matplotlib.pyplot as plt

df1_train = pd.read_csv("./result/DDQN_gamma_0.99_epsilon_1.0_train_data.csv", index_col='episode')

ax = df1_train.plot(title="The Reward for Each Training Episode", fontsize=12, color=['b','r'])
ax.set_xlabel("Episode")
ax.set_ylabel("Reward")
ax.axhline(y=200, color='k', linestyle='--')
plt.show()

df1_test = pd.read_csv("./result/DDQN_gamma_0.99_epsilon_1.0_test_data.csv", index_col='episode')

ax1 = df1_test.plot(title="The Reward for Each Test Episode Using Trained Agent", fontsize=12, color=['b','r'])
ax1.set_xlabel("Episode")
ax1.set_ylabel("Reward")
ax1.set_xlim([0, 100])
ax1.axhline(y=200, color='k', linestyle='--')
plt.show()

gamma_train = pd.read_csv("./result/gamma_train.csv", index_col='episode')

ax2 = gamma_train.plot(title="The Effect of Discount Factor (Training Episodes)", fontsize=12)
ax2.set_xlabel("Episode")
ax2.set_ylabel("Averge Reward for Last 100 Episodes")
ax2.axhline(y=200, color='k', linestyle='--')
plt.show()

gamma_test = pd.read_csv("./result/gamma_test.csv", index_col='episode')

ax3 = gamma_test.plot(title="The Effect of Discount Factor (Test Episodes)", fontsize=12)
ax3.set_xlabel("Episode")
ax3.set_ylabel("Averge Reward for Last 100 Episodes")
ax3.set_xlim([0, 100])
ax3.axhline(y=200, color='k', linestyle='--')
plt.show()

epsilon_train = pd.read_csv("./result/epsilon_train.csv", index_col='episode')

ax4 = epsilon_train.plot(title="The Effect of Epsilon (Training Episodes)", fontsize=12)
ax4.set_xlabel("Episode")
ax4.set_ylabel("Averge Reward for Last 100 Episodes")
ax4.axhline(y=200, color='k', linestyle='--')
plt.show()

epsilon_test = pd.read_csv("./result/epsilon_test.csv", index_col='episode')

ax5 = epsilon_test.plot(title="The Effect of Epsilon (Test Episodes)", fontsize=12)
ax5.set_xlabel("Episode")
ax5.set_ylabel("Averge Reward for Last 100 Episodes")
ax5.set_xlim([0, 100])
ax5.axhline(y=200, color='k', linestyle='--')
plt.show()

alpha_train = pd.read_csv("./result/alpha_train.csv", index_col='episode')

ax6 = alpha_train.plot(title="The Effect of Learning Rate (Training Episodes)", fontsize=12)
ax6.set_xlabel("Episode")
ax6.set_ylabel("Averge Reward for Last 100 Episodes")
ax6.axhline(y=200, color='k', linestyle='--')
plt.show()

alpha_test = pd.read_csv("./result/alpha_test.csv", index_col='episode')

ax7 = alpha_test.plot(title="The Effect of Learning Rate (Test Episodes)", fontsize=12)
ax7.set_xlabel("Episode")
ax7.set_ylabel("Averge Reward for Last 100 Episodes")
ax7.set_xlim([0, 100])
ax7.axhline(y=200, color='k', linestyle='--')
plt.show()