#******************************************************************************
# Import Libraries 

import numpy as np
import random
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim

#******************************************************************************
# Import internal Libraries 

from model import QNetwork

#******************************************************************************
# Hyper parameters 

BUFFER_SIZE = int(1e5)# Buffer size
UPDATE_EVERY = 4      # Update frequency of the network
BATCH_SIZE = 64       # Batch size
GAMMA = 0.99          # Discount factor
TAU = 1e-3            # Soft update of target parameters (interpolation parameter)
LR = 5e-4             # learning rate 

#******************************************************************************
# GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# DQN agent
class DqnAgent():

    def __init__(self, state_size, action_size, seed, fc1_size=64, fc2_size=64, tau=1e-3):
        """
        ****************************************************           
                    Parameters for DQN Agent
        ****************************************************
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_size (int): First fully-connected layer size
            fc2_size (int): Second fully-connected layer size
            tau (float): interpolation parameter 
        ****************************************************
        """ 
        self.tau = tau        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        # Pair of Q-Networks: local and target
        # A target network is a copy of the action-value function (or Q-function)
        # that is held constant to serve as a stable target for learning for some
        # fixed number of timesteps.
        self.qnetwork_local = QNetwork(state_size, action_size, seed, fc1_size, fc2_size).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed, fc1_size, fc2_size).to(device)
        # Choice of optimizer is Adam
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        # Replay memory with buffer size and batch size as parameter
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Time step intiliazation
        self.t_step = 0
        
    def tau(self):
        # Returns TAU
        return self.tau
   
    def step(self, state, action, reward, next_state, done):
        # Save experience in memory
        self.memory.add(state, action, reward, next_state, done)       
        # Time step frequency learning
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # Sample memory if memory is more than our paramters Batch size
            # And learn with a discount factor of GAMMA
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
                
    def act(self, state, epsilon=0.):
        """
        **********************************************************          
              Actions returns per policy for given state
        **********************************************************
            state: current state
            epsilon (float): epsilon-greedy action selection
        **********************************************************
        """ 
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        # Epsilon action selection
        # If number is greater than epsilon, then the agent will exploit
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

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
        with torch.no_grad():
            Q_targets_next, _ = torch.max(self.qnetwork_target(next_states), dim=1)
            Q_targets = rewards + gamma * Q_targets_next.unsqueeze(1) * (1 - dones)
        self.qnetwork_local.train()
        Q_expected = torch.gather(self.qnetwork_local(states), dim=1, index=actions)
        # Mean squared error (MSE) loss calculation        
        loss = F.mse_loss(Q_expected, Q_targets)
        #  Minimize the loss & Clear out the gradients of all variables        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Target network update
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     
## OR
#        states, actions, rewards, next_states, dones = experiences
#        # Q targets for next state
#        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
#        # Q targets for current state
#        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
#        # Expected Q values from local model
#        Q_expected = self.qnetwork_local(states).gather(1, actions)
#        # Mean squared error (MSE) loss calculation
#        loss = F.mse_loss(Q_expected, Q_targets)
#        # Minimize the loss & Clear out the gradients of all variables 
#        self.optimizer.zero_grad()
#        loss.backward()
#        self.optimizer.step()
#        # Target network update
#        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)  

    def soft_update(self, local_model, target_model, tau):
        """
        **********************************************************          
                Soft update model parameters with TAU
                θ_target = τ*θ_local + (1 - τ)*θ_target
        **********************************************************
            local_model : weights from
            target_model: weights to
            tau (float): interpolation parameter 
        **********************************************************
        """ 
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

# Experience storage as tuples
class ReplayBuffer:

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """
        **********************************************************          
                            Replay Buffer
        **********************************************************
            action_size (int): Dimension of each action
            buffer_size (int): Buffer size
            batch_size (int):  Size of training batch
            seed (int): random seed
        **********************************************************
        """ 
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        # Add experience
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        # Sample experience randomly
        experiences = random.sample(self.memory, k=self.batch_size)
        # Vertical stack states, actions, rewards, next_states, dones
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        # Memory size
        return len(self.memory)
    