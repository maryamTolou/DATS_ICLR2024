import torch.nn as nn
import torch


class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()
        self.activation_fn = nn.Tanh()
        self.net = nn.ModuleList([])
        self.num_layers = 7
        self.in_dim = 4
        self.out_dim = 4
        self.hidden_dim = 32
        self.net.append(torch.nn.Linear(self.in_dim, self.hidden_dim))
        for i in range(self.num_layers-2):
            self.net.append(torch.nn.Linear(self.hidden_dim, self.hidden_dim))
        self.net.append(torch.nn.Linear(self.hidden_dim, self.out_dim))
        self.net = nn.Sequential(*self.net)

    def set_params(self, net_weight, net_bias):
        for i in range(self.num_layers):
            del self.net[i].weight
            del self.net[i].bias
        self.net[0].weight = net_weight[:self.hidden_dim * self.in_dim].view([self.hidden_dim, self.in_dim])
        for i in range(self.num_layers - 2):
            self.net[i + 1].weight = net_weight[self.hidden_dim * self.hidden_dim * i + self.hidden_dim * self.in_dim:self.hidden_dim * self.hidden_dim * (i + 1) + self.hidden_dim * self.in_dim].view([self.hidden_dim, self.hidden_dim])
        self.net[self.num_layers - 1].weight = net_weight[self.hidden_dim * self.hidden_dim * (self.num_layers - 3) + self.hidden_dim * self.in_dim:self.hidden_dim * self.hidden_dim * (self.num_layers - 3) + self.hidden_dim * self.in_dim + self.hidden_dim*self.out_dim].view([self.out_dim, self.hidden_dim])
        self.net[0].bias = net_bias[:self.hidden_dim]
        for i in range(self.num_layers - 2):
            self.net[i + 1].bias = net_bias[self.hidden_dim * i + self.hidden_dim:self.hidden_dim * i + self.hidden_dim + self.hidden_dim]
        self.net[self.num_layers - 1].bias = net_bias[(self.num_layers - 3) * self.hidden_dim + self.hidden_dim:(self.num_layers - 3) * self.hidden_dim + self.hidden_dim + self.out_dim]

    def forward(self, x, y, z, t, lowb, upb):
        X = torch.cat((x, y, z, t), dim=1)
        X = 2.0*(X - lowb) / (upb-lowb) - 1.0
        X = self.net[0](X)
        X = self.activation_fn(X)
        for i in range(self.num_layers - 2):
            X = self.net[i + 1](X)
            X = self.activation_fn(X)
        out = self.net[self.num_layers - 1](X)
        return out

class Hypernet(nn.Module):
    def __init__(self):
        super(Hypernet, self).__init__()
        self.in_dim = 4
        self.out_dim = 4
        self.num_layers = 7
        self.hidden_dim = 32
        self.total_net_weight = self.in_dim * self.hidden_dim + (self.num_layers - 2) * self.hidden_dim * self.hidden_dim + self.hidden_dim * self.out_dim
        self.total_net_bias = self.hidden_dim + (self.num_layers - 2) * self.hidden_dim + self.out_dim
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, self.total_net_weight + self.total_net_bias)
        )

    def forward(self, x):
        params = self.net(x)
        net_weight = params[:self.total_net_weight]
        net_bias = params[self.total_net_weight:self.total_net_weight + self.total_net_bias]
        return net_weight, net_bias