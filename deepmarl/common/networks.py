import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Weight initialization similar to DDPG algorithm (Lowe et al.)
    """
    def __init__(self, lr, input_shape, output_shape, discrete, constrained, device, num_units, norm_in=True):
        super(MLP, self).__init__()
        self.lr = lr
        self.input_shape = input_shape
        self.fc1_dims = num_units
        self.fc2_dims = num_units
        self.output_shape = output_shape
        self.device = device

        if norm_in:  # Batch normalization of inputs
            self.in_fn = nn.BatchNorm1d(*self.input_shape)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x

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

        if not discrete and constrained:
            self.activation = T.tanh
        else:
            self.activation = lambda x: x

        self.to(device, dtype=T.double)

    def forward(self, state):
        x = self.in_fn(state)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.mu(x)
        return self.activation(x)
