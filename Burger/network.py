import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import sys
from utils.grid_sample import *


class Base(nn.Module):
    def __init__(self, hparams):
        super(Base, self).__init__()
        # parameters
        self.model_type = hparams.model_type
        self.num_layers = hparams.num_layers
        self.problem = 'forward'
        self.hidden_dim = hparams.hidden_dim
        self.out_dim = hparams.out_dim
        self.in_dim = hparams.in_dim
        self.activation_fn = nn.Tanh()
        if self.num_layers == 0:
            return
        self.net = nn.ModuleList([])
        if self.model_type == 'mad-pinn':
            self.latent_size = hparams.latent_size
            input_dim = self.in_dim + self.latent_size
            if self.num_layers < 2:
                self.net.append(self.activation_fn)
                self.net.append(torch.nn.Linear(input_dim, self.out_dim))
            else:
                self.net.append(torch.nn.Linear(input_dim, self.hidden_dim))
                self.net.append(self.activation_fn)
                for i in range(self.num_layers-2):
                    self.net.append(torch.nn.Linear(self.hidden_dim, self.hidden_dim))
                    self.net.append(self.activation_fn)
                self.net.append(torch.nn.Linear(self.hidden_dim, self.out_dim))
            self.net = nn.Sequential(*self.net)

            # self.N_way = hparams.sampler['n_way']
            latent_init = torch.randn([len(hparams.params), self.latent_size]) / torch.sqrt(torch.tensor(self.latent_size))
            self.latent_vector = torch.nn.Parameter(latent_init, requires_grad=True)

            
        elif self.model_type == 'hyper-pinn':
            self.latent_size = None
            input_dim = self.in_dim
            self.net.append(torch.nn.Linear(input_dim, self.hidden_dim))
            for i in range(self.num_layers - 2):
                self.net.append(torch.nn.Linear(self.hidden_dim, self.hidden_dim))
            self.net.append(torch.nn.Linear(self.hidden_dim, self.out_dim))

    def set_weight(self, net_weight, net_bias):
        for i in range(self.num_layers):
            del self.net[i].weight
            del self.net[i].bias
        self.net[0].weight = net_weight[:16].view([8, 2])
        for i in range(self.num_layers-2):
            self.net[i+1].weight = net_weight[64*i+16:64*i+16+64].view([8, 8])
        self.net[self.num_layers-1].weight = net_weight[64*5+16:64*5+16+8].view([1, 8])

        self.net[0].bias = net_bias[:8]
        for i in range(self.num_layers-2):
            self.net[i+1].bias = net_bias[8*i+8:8*i+8+8]
        self.net[self.num_layers-1].bias = net_bias[5*8+8:5*8+8+1]

    def forward(self, x, latent_vector):

        if self.model_type == 'hyper-pinn':
            x = self.net[0](x)
            x = self.activation_fn(x)
            for i in range(self.num_layers-2):
                x = self.net[i+1](x)
                x = self.activation_fn(x)
            out = self.net[self.num_layers-1](x)
        
        elif self.model_type == 'mad-pinn':
            latent_vector = latent_vector.view(1, -1)
            latent_vector = latent_vector.repeat(x.shape[0], 1)
            x = torch.cat((x, latent_vector), dim=1)
            out = self.net(x)
        return out


class Hypernet(nn.Module):
    def __init__(self, hparams):
        super(Hypernet, self).__init__()
        self.num_layers = hparams.num_layers
        self.hidden_dim = hparams.hidden_dim
        self.out_dim = hparams.out_dim
        self.in_dim = hparams.in_dim
        self.total_net_weight = self.in_dim*self.hidden_dim + (self.num_layers-2)*self.hidden_dim*self.hidden_dim + self.hidden_dim*self.out_dim
        self.total_net_bias = self.hidden_dim + (self.num_layers-2)*self.hidden_dim + self.out_dim
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, self.total_net_weight+self.total_net_bias)
        )
        
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.constant_(layer.bias, 0.05)

    def forward(self, x):
        x = x * 10
        params = self.net(x)
        net_weight = params[:self.total_net_weight]
        net_bias = params[self.total_net_weight:self.total_net_weight+self.total_net_bias]
        return net_weight, net_bias
    

class SelectNet(nn.Module):
    def __init__(self, hparams):
        super(SelectNet, self).__init__()
        self.hidden_dim = hparams.selectnet['hidden_dim']

        self.net = nn.Sequential(
            nn.Linear(1, self.hidden_dim[0]),
            nn.Tanh(),
            nn.Linear(self.hidden_dim[0], 1),
            nn.Sigmoid()
        )
        
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        v = self.net(x)

        return v