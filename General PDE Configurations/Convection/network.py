import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


class Base(nn.Module):
    def __init__(self, hparams):
        super(Base, self).__init__()
        # parameters
        self.model_type = hparams.model_type
        self.in_dim = hparams.network['in_dim']
        self.out_dim = hparams.network['out_dim']
        self.num_layers = hparams.network['num_layers']
        self.hidden_dim = hparams.network['hidden_dim']
        self.activation_fn = nn.Tanh()
        if self.num_layers == 0:
            return
        self.net = nn.ModuleList([])
        if self.model_type == 'mad-pinn':
            self.latent_size = hparams.network['latent_size']
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
            latent_init = torch.randn([hparams.sampler['n_way'], self.latent_size]) / torch.sqrt(torch.tensor(self.latent_size))
            self.latent_vector = torch.nn.Parameter(latent_init, requires_grad=True)

            
        elif self.model_type == 'hyper-pinn':
            self.latent_size = None
            input_dim = self.in_dim
            self.net.append(torch.nn.Linear(input_dim, self.hidden_dim))
            for i in range(self.num_layers - 2):
                self.net.append(torch.nn.Linear(self.hidden_dim, self.hidden_dim))
            self.net.append(torch.nn.Linear(self.hidden_dim, self.out_dim))

    def set_params(self, net_weight, net_bias):
        for i in range(self.num_layers):
            del self.net[i].weight
            del self.net[i].bias
        self.net[0].weight = net_weight[:self.hidden_dim*2].view([self.hidden_dim, 2])
        for i in range(self.num_layers-2):
            self.net[i+1].weight = net_weight[self.hidden_dim*self.hidden_dim*i+self.hidden_dim*2:self.hidden_dim*self.hidden_dim*(i+1)+self.hidden_dim*2].view([self.hidden_dim, self.hidden_dim])
        self.net[self.num_layers-1].weight = net_weight[self.hidden_dim*self.hidden_dim*(self.num_layers-3)+self.hidden_dim*2:self.hidden_dim*self.hidden_dim*(self.num_layers-3)+self.hidden_dim*2+self.hidden_dim].view([1, self.hidden_dim])

        self.net[0].bias = net_bias[:self.hidden_dim]
        for i in range(self.num_layers-2):
            self.net[i+1].bias = net_bias[self.hidden_dim*i+self.hidden_dim:self.hidden_dim*i+self.hidden_dim+self.hidden_dim]
        self.net[self.num_layers-1].bias = net_bias[(self.num_layers-3)*self.hidden_dim+self.hidden_dim:(self.num_layers-3)*self.hidden_dim+self.hidden_dim+1]

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
        # out = self.net(x)
        return out


class Hypernet(nn.Module):
    def __init__(self, hparams):
        super(Hypernet, self).__init__()
        self.in_dim = hparams.network['in_dim']
        self.out_dim = hparams.network['out_dim']
        self.num_layers = hparams.network['num_layers']
        self.hidden_dim = hparams.network['hidden_dim']
        self.total_net_weight = self.in_dim*self.hidden_dim + (self.num_layers-2)*self.hidden_dim*self.hidden_dim + self.hidden_dim*self.out_dim
        self.total_net_bias = self.hidden_dim + (self.num_layers-2)*self.hidden_dim + self.out_dim
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, self.total_net_weight+self.total_net_bias)
        )

    def forward(self, x):
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