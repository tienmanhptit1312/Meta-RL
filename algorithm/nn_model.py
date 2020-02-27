import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.distribution import Normal


LOG_SIX_MAX = 2
LOG_SIX_MIN = -20
epsilon = 1e-6

def weight_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)

class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dims, activation):
        super(ValueNetwork, self).__init__()

        self.model = nn.ModuleList()
        in_hidden_dim = num_inputs
        for hidden_dim in hidden_dims:
            layer = nn.Linear(in_hidden_dim, hidden_dim)
            torch.nn.init.xavier_uniform_(layer.weight, gain=1)
            torch.nn.init.constant_(layer.bias, 0)
            self.model.append(layer)
            in_hidden_dim = hidden_dim
        self.activattion = activattion
        
    def forward(self, input):
        h = input 
        for layer in self.model:
            h = layer(h)
            h = self.activation
        return h


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dims, activation):

        super(QNetwork, self).__init__()

        self.Q1 = nn.ModuleList()
        self.Q2 = nn.ModuleList()
        self.activation = activation
        in_hidden_dim = num_inputs + num_actions

        for hidden_dim in hidden_dims:
            layer = nn.Linear(in_hidden_dim, hidden_dim)
            torch.nn.init.xavier_uniform_(layer.weight, gain=1)
            torch.nn.init.constant_(layer.bias, 0)

            self.Q1.append(layer)
            self.Q2.append(layer)
            in_hidden_dim = hidden_dim

    def forward(self, input, action):
        h1 = torch.cat([input, action], dim=1)
        h2 = torch.cat([input, action], dim=1)

        for (layer1, layer2) in zip(self.Q1, self.Q2):
            h1 = layer1(h1)
            h1 = self.activation(h1)
            h2 = layer2(h2)
            h2 = self.activation(h2)

        return h1, h2

class GaussianPolicy(nn.Module):

    def __init__(self, num_inputs, num_action, hidden_dims, action_space=None):

        super(GaussianPolicy, self).__init__()

        self.policy = nn.ModuleList()
        in_hidden = num_inputs
        for hidden_dim in hidden_dims:

            layer = nn.Linear(in_hidden, hidden_dim)
            torch.nn.init.xavier_uniform_(layer.weight, gain=1)
            torch.nn.init.constant_(layer.bias, 0)

            self.policy.append(layer)
            in_hidden = hidden_dim

        self.mean_actions = nn.Linear(in_hidden, num_actions)
        torch.nn.init.xavier_uniform_(self.mean_actions.weight, gain=1)
        torch.nn.init.constant_(self.mean_actions.bias, 0)

        self.std_actions = nn.Linear(in_hidden, num_actions)
        torch.nn.init.xavier_uniform_(self.std_actions.weight, gain=1)
        torch.nn.init.constant_(self.std_actions.bias, 0)

        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.0)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.0)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.0)

    def forward(self, input):
        h = input

        for layer in self.policy:
            h = layer(h)
            h = torch.nn.ReLU(h)
        
        mean = self.mean_actions(h)
        log_std = self.std_actions(h)
        # clip value of log std
        log_std = torch.clamp(log_std, min=LOG_SIX_MIN, max=LOG_SIX_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)

        std = log_std.exp()
        normal = Normal(mean, std)

        x_t = normal.rsample() # for reparameterization trick (mean + std*N(0,1))
        y_t = torch.tanh(x_t)

        action = y_t*self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing action bound
        log_prob -= torch.log(self.action_scale*(1 - y_t.pow(2) + epsilon))
        log_prob = log_prob.sum(1, keepdim=True)

        mean = torch.tanh(mean)*self.action_scale + self.action_bias
        return acction, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device) 

class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        
        self.policy = nn.MoudleList()
        in_hidden = num_inputs
        for hidden_dim in hidden_dims:
            layer = nn.Linear(in_hidden, hidden_dim)
            # torch.nn.init.xavier_uniform_(layer.weight, gain=1)
            # torch.nn.init.constant_(layer.bias, 0)

            self.policy.append(layer)
            in_hidden = hidden_dim

        self.mean = nn.Linear(in_hidden, num_actions)
        self.noise = nn.Tensor(num_actions)

        self.apply(weight_init_)
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        h = state
        for layer in self.policy:
            h = layer(h)
            h = nn.ReLU(h)

        mean = nn.tanh(self.mean(h))*self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)

        noise = self.noise.normal_(0, std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)