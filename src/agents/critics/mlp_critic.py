import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLP_Critic(nn.Module):
    def __init__(self, input_dim, output_dim, device, norm_in=True, fc1_dim=64, fc2_dim=64, weight_init=True):
        super().__init__()

        self.device = device

        if norm_in: # Batch normalization of observations
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x

        self.fc1 = nn.Linear(input_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.out = nn.Linear(fc2_dim, output_dim)

        if weight_init:
            # This weight initialization scheme is the same one suggested
            # in Lilicrap et al https://arxiv.org/abs/1509.02971
            f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
            T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
            T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)

            f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
            T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
            T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)

            T.nn.init.uniform_(self.out.weight.data, -3e-3, 3e-3)
            T.nn.init.uniform_(self.out.bias.data, -3e-3, 3e-3)

    def forward(self, obs):
        if not isinstance(obs, T.Tensor):
            x = T.tensor(obs, dtype=T.float64, device=self.device)
        else:
            x = obs
        x = self.in_fn(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.out(x)
