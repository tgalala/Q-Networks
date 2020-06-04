#******************************************************************************
# Import Libraries 

import torch
import torch.nn as nn
import torch.nn.functional as F

#******************************************************************************
# DQN Architecture
#******************************************************************************
class QNetwork(nn.Module):
    
    def __init__(self, state_size, action_size, seed, fc1_size = 64, fc2_size = 64):
        """
        ***********************************************           
                 Parameters for Model Building
        ***********************************************
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_size (int): First fully-connected layer size
            fc2_size (int): Second fully-connected layer size
        ***********************************************
        """      
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.out = nn.Linear(fc2_size, action_size)     

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action = self.out(x)
        return action

#******************************************************************************
# Dueling Architecture  
#******************************************************************************        
class DuelingQNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed, fc1_size = 64, fc2_size = 64):
        """
        ***********************************************           
                 Parameters for Model Building
        ***********************************************
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_size (int): First fully-connected layer size
            fc2_size (int): Second fully-connected layer size
        ***********************************************
        """      
        super(DuelingQNetwork, self).__init__()
        self.num_actions = action_size
        fc3_1_size = fc3_2_size = 32
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        ## Two seperate splits
        # One that calculate V(s)
        self.fc3_1 = nn.Linear(fc2_size, fc3_1_size)
        self.fc4_1 = nn.Linear(fc3_1_size, 1)
        # One that calculate A(s,a)
        self.fc3_2 = nn.Linear(fc2_size, fc3_2_size)
        self.fc4_2 = nn.Linear(fc3_2_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        val = F.relu(self.fc3_1(x))
        val = self.fc4_1(val)        
        adv = F.relu(self.fc3_2(x))
        adv = self.fc4_2(adv)
        action = val + adv - adv.mean(1).unsqueeze(1).expand(state.size(0), self.num_actions)
        return action
    