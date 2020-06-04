#******************************************************************************
# Import Libraries 

import numpy as np
import torch
import torch.nn.functional as F

#******************************************************************************
# Import internal Libraries 

from dqn_agent import DqnAgent

#******************************************************************************
# Double DQN is based on David Silver et al. (2016)
# Deep reinforcement learning with double Q-Learning.

# GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Double DQN agent
class doubleDqnAgent(DqnAgent):
    def __init__(self, state_size, action_size, seed, fc1_size=64, fc2_size=64, tau=1e-3):
        """
        ****************************************************           
                 Parameters for Double DQN Agent
        ****************************************************
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_size (int): First fully-connected layer size
            fc2_size (int): Second fully-connected layer size
        ****************************************************
        """ 
        super(doubleDqnAgent, self).__init__(state_size, action_size, seed, fc1_size, fc2_size, tau)
        self.tau = tau
    
    def learn(self, experiences, gamma):
        """
        ***************************************************************          
                    Updates parameters based on experience
        ***************************************************************
            experiences (Tuple of tensors): (s, a, r, s', done) 
            gamma (float): Discount factor
        ***************************************************************
        """         
        # done(s) flag dt tells us if we should use the bootstrapped estimate of our target
        states, actions, rewards, next_states, dones = experiences        
        self.qnetwork_target.eval()
        self.qnetwork_local.eval()
        with torch.no_grad():
            _, Q_max_action = torch.max(self.qnetwork_local(next_states), dim=1) 
            Q_next = self.qnetwork_target(next_states)
            Q_targets_next = torch.gather(Q_next, dim=1, index=Q_max_action.unsqueeze(1))
            Q_targets = rewards + gamma * Q_targets_next * (1 - dones) 
        self.qnetwork_target.train()
        self.qnetwork_local.train()
        Q_expected = torch.gather(self.qnetwork_local(states), dim=1, index=actions)
        # Mean squared error (MSE) loss calculation        
        loss = F.mse_loss(Q_expected, Q_targets)
        #  Minimize the loss & Clear out the gradients of all variables        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Target network update
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)