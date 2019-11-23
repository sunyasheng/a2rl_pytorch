import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def build_mlp(dims, activation=nn.ReLU):
    layers = []
    for (in_dim, out_dim) in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(in_dim, out_dim), activation()]
    
    return nn.Sequential(*layers)#.apply(weights_init)

def weights_init(m):
    if hasattr(m, 'weight'):
        nn.init.xavier_uniform_(m.weight)

class ActorCritic(nn.Module):
    def __init__(self, args, hidden_size=128, dims=[128, 128]):
        super(ActorCritic, self).__init__()
        self.ob_dim = args.ob_dim
        self.act_dim = args.act_dim
        self.dims = dims
        self.hidden_size = hidden_size
        self.dims = [args.ob_dim] + dims + [self.hidden_size]

        self.mlp = build_mlp(self.dims)

        self.lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)

        self.critic_linear = nn.Linear(self.hidden_size, 1)
        self.actor_linear = nn.Linear(self.hidden_size, self.act_dim)

        self.apply(weights_init)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        
        self.train()
    
    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = self.mlp(inputs)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        return self.critic_linear(x), self.actor_linear(x), (hx, cx)