import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class MLP(nn.Module):
    def __init__(self, lr, input_shape, output_shape, activation, device, num_units):

        super(MLP, self).__init__()
        self.lr = lr
        self.input_shape = input_shape
        self.fc1_dims = num_units
        self.fc2_dims = num_units
        self.output_shape = output_shape
        self.device = device

        self.fc1 = nn.Linear(*self.input_shape, self.fc1_dims)
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)

        f3 = 0.003
        self.mu = nn.Linear(self.fc2_dims, *self.output_shape)
        T.nn.init.uniform_(self.mu.weight.data, -f3, f3)
        T.nn.init.uniform_(self.mu.bias.data, -f3, f3)

        self.activation = activation

        self.to(device)

    def forward(self, state):

        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.activation(self.mu(x))

        return x


