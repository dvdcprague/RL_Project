"""
DDQN
"""

import numpy as np
import random as rand
from collections import deque
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras import backend as K

class DQNLearner(object):

    def __init__(self, \
        state_dim=8, \
        action_dim = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 1.0, \
        radr = 0.99, \
        units = 40, \
        update_target_freq=600,\
        verbose = False):

        self.verbose = verbose
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.units = units
        self.update_target_freq = update_target_freq
        
        self.replay_size = 500000
        self.batch_size = 32
        self.minisamples = 1000
        self.D = deque()
        
        self.build_nn()
        self.step = 0
    
    def build_nn(self):
        self.model = self.build_model()
        self.target_model = self.build_model()
        
    def huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)
       
    def build_model(self):
        model = Sequential()
        model.add(Dense(units=self.units, input_shape=(self.state_dim,)))
        model.add(Activation('relu'))
        model.add(Dense(units=self.units))
        model.add(Activation('relu'))
        model.add(Dense(self.action_dim))
        model.add(Activation('linear'))
        
        model.compile(loss=self.huber_loss, optimizer=Adam(lr=self.alpha))
        
        return model
        
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        

    def epsilon_greedy_action(self, state):
   
        state = np.array([state])
        if rand.uniform(0.0, 1.0) <= self.rar:
            action = rand.randint(0,self.action_dim-1)
        else:
            q = self.model.predict(state)[0]
            action = np.argmax(q)
        
        return action
        
    def set_action(self, state):
   
        state = np.array([state])
        q = self.model.predict(state)[0]
        action = np.argmax(q)
        
        return action
        
        
    def store_experience(self, state, action, reward, next_state, done):
        #store the experience in replay memory
        state = np.array([state])
        next_state = np.array([next_state])
        self.D.append((state, action, reward, next_state, done))
        if len(self.D) > self.replay_size:
            self.D.popleft()
            
        if done:
            self.rar = self.rar * self.radr
            
        if self.step % self.update_target_freq == 0:
            self.update_target_model()
        
        self.step += 1
        

    def train(self):
#        inputs = np.zeros((self.batch_size, self.state_dim))
#        targets = np.zeros((self.batch_size, self.action_dim)) 
        
        if len(self.D) <= self.minisamples:
            return 0.0
        
        batch = rand.sample(self.D, self.batch_size)

        state=[data[0][0] for data in batch]
        action=[data[1] for data in batch]
        reward=[data[2] for data in batch]
        next_state=[data[3][0] for data in batch]
        done = [data[4] for data in batch]
        
        state  = np.array(state)
        next_state = np.array(next_state)
 
        q  = self.model.predict(state)
        q_eval = self.model.predict(next_state)
        q1 = self.target_model.predict(next_state)
 
        targets = np.zeros((self.batch_size, self.action_dim))

        for i in range(0, len(batch)):
            r = reward[i]
            a = action[i]
            target = q[i]
 
            target_for_action = r 
            if not done[i]:
                target_for_action += self.gamma * q1[i][np.argmax(q_eval[i])]
            target[a] = target_for_action
            targets[i, :] = target
                

        loss = self.model.train_on_batch(state, targets)
        return loss

    

if __name__=="__main__":
    print("Remember Q from Star Trek? Well, this isn't him")
