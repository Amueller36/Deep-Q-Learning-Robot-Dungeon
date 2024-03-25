import os

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DeepQNetwork(nn.Module):
    def __init__(self, learning_rate, n_actions, name, input_dims, chkpt_dir):
        super().__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        size_of_input_data = input_dims[0] # ist die Anzahl der Features, die wir in den Input legen. Shape ist (172, 0 )
        self.fc1 = nn.Linear(in_features=size_of_input_data, out_features=512, dtype=T.float32)
        self.fc2 = nn.Linear(in_features=512, out_features=256, dtype=T.float32)
        self.fc3 = nn.Linear(in_features=256, out_features=128, dtype=T.float32)
        self.fc4 = nn.Linear(in_features=128, out_features=n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        flat1 = F.relu(self.fc1(state))
        flat2 = F.relu(self.fc2(flat1))
        flat3 = F.relu(self.fc3(flat2))
        actions = self.fc4(flat3)
        return actions

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))
