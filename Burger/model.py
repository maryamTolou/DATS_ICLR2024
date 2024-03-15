import numpy as np
import torch
import torch.optim as optim
import os
import sys
from matplotlib import rc
from tqdm import tqdm
from Burger  import Equation, Test, Load_ground_truth, Generate_train_data
import utils.lr as utr
import network as network
import torch.nn as nn
import time
import json
import utils.plot as utp
import matplotlib.pyplot as plt
import glob
import random
import re
import shutil
import math


def pde_test(pde, task, t_test, x_test, u_test, lambda_1, net_u_2d, problem, it, loss_list, output_path, tag, epochs, latent_vector, device):
    l2_loss, mse_loss = Test(pde, task, t_test, x_test, u_test, lambda_1, net_u_2d, problem, it, loss_list, output_path, tag, epochs, latent_vector, device)
    
    return l2_loss, mse_loss


class MODEL():
    def __init__(self, base_network, device, model_type, hypernet, json_path, hparams, PDE):
        
        self.device = device
        self.hparams = hparams
        self.model_type = model_type
        self.tag = model_type + '_' + hparams.sampler['type'] + '_' + str(hparams.id) + '_' + str(hparams.seed) + '_num_' + str(hparams.num_train) 
        if model_type == 'mad-pinn':
            self.output_path = "results_burger_mad/figures/{}".format(self.tag)
        else:
            self.output_path = "results_burger_hyper/figures/{}".format(self.tag)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        if not os.path.exists(self.output_path + '/epoch_figs'):
            os.makedirs(self.output_path + '/epoch_figs')
        shutil.copy(json_path, self.output_path)
        self.dnn = base_network
        self.hypernet = hypernet
        # random sampling at every iteration
        self.random_f = hparams.random_f
        # number of points
        self.num_train = hparams.num_train
        self.num_ic = hparams.num_ic
        self.num_bc = hparams.num_bc
        
        self.num_val_res = hparams.validating['num_res']
        self.num_ic_val = hparams.validating['num_ic']
        self.num_bc_val = hparams.validating['num_bc']
    
        self.f_scale = hparams.f_scale
        self.use_cell = hparams.use_cell
        self.fine_tune_iter = hparams.fine_tune_iter
        if self.model_type == 'mad-pinn':
            self.latent_vector = self.dnn.latent_vector
            self.latent_id = {}
            self.latent_reg = hparams.latent_reg
        else:
            self.latent_vector = None
            self.latent_reg = None

        self.optim = hparams.optim
        self.first_call = True
        self.lr = hparams.lr
        self.max_iter = hparams.max_iter
        self.set_optimizer()

        self.iter = 0
        self.loss_list = []
        self.pde = PDE
        self.problem = 'forward'
        self.test_nus = hparams.test_nus
        self.nus = hparams.params
   
        self.t_train_f, self.x_train_f, self.t_train, self.x_train, self.u_train = Generate_train_data(
            self.num_train, self.num_ic, self.num_bc, self.device)
        
        self.task_budget = {task: [self.num_train] for task in self.nus}
        
        self.probs = {}
        self.sampler = hparams.sampler
        for task in self.nus:
            self.probs[task] = torch.ones((1, 1))*(1/len(self.nus))
        
        self.sampling_type = hparams.sampler['type']
        self.lr_q = self.sampler['eta']
        self.residual_sampling = hparams.sampler['residual_sampling']
        if self.sampling_type == 'uniform':
             self.residual_sampling = False
        self.budget = hparams.num_train  * len(self.nus)
        self.kl_type = hparams.sampler['kl']
        self.epsilon = hparams.sampler['epsilon']
        self.prob_period = hparams.sampler['period']
        # self.n_way = hparams.sampler['n_way']

        self.eta = hparams.sampler['eta']        
        self.t_val, self.x_val, self.t_val_init, self.x_val_init, self.u_val_init = Generate_train_data(
            self.num_val_res, self.num_ic_val, self.num_bc_val, self.device)


         # for plotting
        self.train_task_init_loss = {task: [] for task in self.nus}
        self.train_task_pde_loss = {task: [] for task in self.nus}
        self.train_task_total_loss = {task: [] for task in self.nus}

        self.l2_loss = {task: [] for task in self.nus}
        self.mse_loss = {task: [] for task in self.nus}

        self.prob_per_epoch = {task: [] for task in self.nus}
        self.gi_gsum_per_epoch = {task: [] for task in self.nus}

        self.train_task_init_loss_val = {}
        self.train_task_pde_loss_val = {}
        self.train_task_total_loss_val = {}
        self.epoch_dataset = {}

        self.scaled_loss_total = torch.tensor([0.0]).to(self.device)
        self.u_loss_total = torch.tensor([0.0]).to(self.device)
        self.f_loss_total = torch.tensor([0.0]).to(self.device)
        self.av_l2_loss = 10000

        self.pace_k = 1000
        self.selectnet = {}
        self.loss_t_1 = {}
        for task in self.nus:
            self.selectnet[task] = nn.Parameter(torch.tensor(1.0, requires_grad=True).float().to(self.device))
        self.select_optimizer = optim.Adam(self.selectnet.values(), lr=1e-2)
        self.pace_k_step = (self.pace_k) / ((self.max_iter - 4000) // 100)
        self.ps = {}
                
        if  self.residual_sampling is False and self.hparams.residual_sampler['type'] == 'rad' and self.hparams.residual_sampler['adaptive_type'] == 'gradual':
            steps =  (self.max_iter / self.hparams.residual_sampler['period'] - 2)
            self.adaptive_points_no = math.floor(int(self.num_train/2) / steps)
            self.num_train = int(self.num_train/2) + round(steps * (int(self.num_train/2) / steps - self.adaptive_points_no))
            self.t_train_f, self.x_train_f, self.t_train, self.x_train, self.u_train = Generate_train_data(
            self.num_train, self.num_ic, self.num_bc, self.device)
            
        for task in self.nus:
            self.epoch_dataset[task] = {'t_train_f': self.t_train_f, 'x_train_f': self.x_train_f}
            self.task_budget[task][0] = self.num_train
         
    
    def task_sampler(self):
        gradients_i = {}
        gradients_j = {}
        self.gradient_training = {}
        task_losses_i = {}
        task_losses_j = {}
        probs = {}
        task_u_losses = {}
        task_f_losses = {}
      
        for i, task in enumerate(self.nus):
            self.optimizer.zero_grad()
            self.nu = self.nus[i]
            if self.model_type == 'hyper-pinn':
                net_weight, net_bias = self.hypernet(torch.tensor([self.nu]).to(self.device))
                self.dnn.set_weight(net_weight, net_bias)  
                latent = None
            elif self.model_type == 'mad-pinn':
                latent = self.dnn.latent_vector[self.latent_id[task]]
                
            f_pred = self.net_f_2d(self.t_train_f, self.x_train_f, self.pde, latent)
            loss_f = torch.mean(f_pred ** 2)
            u_pred = self.net_u_2d(self.t_train, self.x_train, self.pde, latent)
            loss_u = torch.mean((self.u_train - u_pred) ** 2)
            
            if self.model_type == 'hyper-pinn':
                scaled_loss = loss_u + self.f_scale * loss_f
                scaled_loss.backward(retain_graph=True)
                gradient_ = [param.grad.detach().clone() for param in self.hypernet.parameters()]
                self.gradient_training[task] = gradient_
            elif self.model_type == 'mad-pinn':
                loss_reg = torch.mean(torch.square(self.latent_vector))
                scaled_loss = loss_u + self.f_scale * loss_f + self.latent_reg * loss_reg
                scaled_loss.backward(retain_graph=True)
                gradient_ = [param.grad.detach().clone() for param in self.dnn.parameters()]
                self.gradient_training[task] = gradient_

            else:
                NotImplementedError
                return 0
            

            task_losses_i[task] = scaled_loss.detach()
            task_u_losses[task] = loss_u.detach()
            task_f_losses[task] = loss_f.detach()
            
            if self.iter % self.prob_period == 0:            
                gradient = []
                for g in gradient_:
                    if g is not None:
                        gradient.append(g.detach().view(-1))  # <-- detach added here
                gradient = torch.cat(gradient)
                gradients_i[task] = gradient
                
                
                self.optimizer.zero_grad()
                if self.model_type == 'hyper-pinn':
                    net_weight, net_bias = self.hypernet(torch.tensor([self.nu]).to(self.device))
                    self.dnn.set_weight(net_weight, net_bias)  
                    latent = None
                elif self.model_type == 'mad-pinn':
                    latent = self.dnn.latent_vector[self.latent_id[task]]
                    
                f_pred = self.net_f_2d(self.t_val, self.x_val, self.pde, latent)
                loss_f = torch.mean(f_pred ** 2)
                u_pred = self.net_u_2d(self.t_val_init, self.x_val_init, self.pde, latent)
                loss_u = torch.mean((self.u_val_init - u_pred) ** 2)
                
                if self.model_type == 'hyper-pinn':
                    scaled_loss = loss_u + self.f_scale * loss_f
                    gradient_ = torch.autograd.grad(scaled_loss, self.hypernet.parameters(), retain_graph=False, allow_unused=True)
                elif self.model_type == 'mad-pinn':
                    loss_reg = torch.mean(torch.square(self.latent_vector))
                    scaled_loss = loss_u + self.f_scale * loss_f + self.latent_reg * loss_reg
                    gradient_ = torch.autograd.grad(scaled_loss, self.dnn.parameters(), retain_graph=False, allow_unused=True)

                else:
                    NotImplementedError


                
                task_losses_j[task] = scaled_loss.detach()
                gradient = []
                for g in gradient_:
                    if g is not None:
                        gradient.append(g.detach().view(-1))  # <-- detach added here
                gradient = torch.cat(gradient)
                gradients_j[task] = gradient

        if self.iter % self.prob_period == 0:
            for i, task in enumerate(self.nus):
                if self.kl_type == 'uniform':
                    gigsum = torch.sum(torch.stack([torch.sqrt(task_losses_i[task] * task_losses_j[task2]) * torch.nn.functional.cosine_similarity(
                        gradients_i[task], gradients_j[task2], dim=0) for j, task2 in enumerate(self.nus)]))
                    self.gi_gsum_per_epoch[task].append(gigsum.item())
                    probs[task] = 1/len(self.nus) * torch.exp(self.eta * gigsum)
                elif self.kl_type == 'consecutive':
                    gigsum = torch.sum(torch.stack([torch.sqrt(task_losses_i[task] * task_losses_j[task2]) * torch.nn.functional.cosine_similarity(
                            gradients_i[task], gradients_j[task2], dim=0) for j, task2 in enumerate(self.nus)]))
                    self.gi_gsum_per_epoch[task].append(gigsum.item())
                    probs[task] = (self.probs[task].to(self.device) * torch.exp(self.eta * gigsum))
                else:
                    print("KL option is not implemented")

            prob_sum = sum(torch.sum(p) for p in probs.values())

            for task in probs.keys():
                self.probs[task] = (probs[task] / prob_sum)

                
        self.gradients_i = gradients_i
        self.task_losses_i = task_losses_i
        self.task_f_losses = task_f_losses
        self.task_u_losses = task_u_losses
            

    def set_optimizer(self):
        if self.model_type == 'hyper-pinn':
            if self.optim == 'lbfgs':
                self.optimizer = optim.LBFGS(
                    self.hypernet.parameters(),
                    lr=self.lr, 
                    line_search_fn="strong_wolfe"       # can be "strong_wolfe"
                )
            elif self.optim == 'adam':
                self.optimizer = optim.Adam(self.hypernet.parameters(), lr=self.lr)
            else:
                raise NotImplementedError()
        elif self.model_type == 'mad-pinn':
            if self.optim == 'lbfgs':
                self.optimizer = optim.LBFGS(
                    self.dnn.parameters(),
                    lr=self.lr, 
                    line_search_fn="strong_wolfe"       # can be "strong_wolfe"
                )
            elif self.optim == 'adam':
                self.optimizer = optim.Adam(self.dnn.parameters(), lr=self.lr)
            else:
                raise NotImplementedError()

        self.StepLR = utr.CosineAnnealingLR(self.optimizer, T_max=20000, warmup_steps=300)

    def net_u_2d(self, t, x, pde, latent_vector):
        if self.use_cell:
            t = t * 2 - 1
            x = torch.cat([t, x], dim=-1).unsqueeze(0).unsqueeze(0)
        else:
            x = torch.cat([t, x], dim=1)
        u = self.dnn(x, latent_vector)
        return u

    def net_f_2d(self, t, x, pde, latent_vector):
        u = self.net_u_2d(t, x, pde, latent_vector)
        f = Equation(u, t, x, self.nu)
        return f

    def loss_func_2d(self):
        if self.model_type == 'hyper-pinn':
            self.hypernet.train()
        self.dnn.train()
        self.scaled_loss_total = torch.tensor([0.0]).to(self.device)
        self.u_loss_total = torch.tensor([0.0]).to(self.device)
        self.f_loss_total = torch.tensor([0.0]).to(self.device)

        scaled_loss = 0
        
        if self.sampling_type == "uniform":
            for i, task in enumerate(self.nus):
                self.t_train_f = self.epoch_dataset[task]['t_train_f']
                self.x_train_f = self.epoch_dataset[task]['x_train_f']
                self.nu = task
                if self.model_type == 'hyper-pinn':
                    net_weight, net_bias = self.hypernet(torch.tensor([self.nu]).to(self.device))
                    self.dnn.set_weight(net_weight, net_bias)  
                    latent = None
                elif self.model_type == 'mad-pinn':
                    if self.first_call is True:
                        self.latent_id[task] = i
                    latent = self.dnn.latent_vector[self.latent_id[task]]
                    
                f_pred = self.net_f_2d(self.t_train_f, self.x_train_f, self.pde, latent)
                loss_f = torch.mean(f_pred ** 2)
                u_pred = self.net_u_2d(self.t_train, self.x_train, self.pde, latent)
                loss_u = torch.mean((self.u_train - u_pred) ** 2)
                
                if self.model_type == 'hyper-pinn':
                    scaled_loss = loss_u + self.f_scale * loss_f
                elif self.model_type == 'mad-pinn':
                    loss_reg = torch.mean(torch.square(self.latent_vector))
                    scaled_loss = loss_u + self.f_scale * loss_f + self.latent_reg * loss_reg

                else:
                    NotImplementedError

                self.train_task_init_loss_val[task] = loss_u
                self.train_task_pde_loss_val[task] = loss_f
                self.train_task_total_loss_val[task] = scaled_loss
                self.prob_per_epoch[task].append(self.probs[task].item())

                self.scaled_loss_total += scaled_loss
                self.u_loss_total += loss_u
                self.f_loss_total += loss_f
            self.scaled_loss_total.backward(retain_graph=True)
        else:
            self.task_sampler()
            self.optimizer.zero_grad()
            for i, task in enumerate(self.nus):
                self.t_train_f = self.epoch_dataset[task]['t_train_f']
                self.x_train_f = self.epoch_dataset[task]['x_train_f']
                self.prob_per_epoch[task].append(self.probs[task].item())
                self.nu = task
                if self.residual_sampling == False:
                    if self.model_type == 'hyper-pinn':
                        for p, g in zip(self.hypernet.parameters(), self.gradient_training[task]):
                            if self.iter>1000:
                                p.grad += self.probs[task].item() * g
                            else:
                                p.grad += g
                            
                    elif self.model_type == 'mad-pinn': #Incomplete!!!!
                        for p, g in zip(self.dnn.parameters(), self.gradient_training[task]):
                            p.grad += self.probs[task].item() * g
                    else:
                        NotImplementedError
                else:
                    if self.model_type == 'hyper-pinn':
                        for p, g in zip(self.hypernet.parameters(), self.gradient_training[task]):
                            p.grad += g
                            
                    elif self.model_type == 'mad-pinn': 
                        for p, g in zip(self.dnn.parameters(), self.gradient_training[task]):
                            p.grad += g
                    else:
                        NotImplementedError

                self.train_task_init_loss_val[task] = self.task_u_losses[task]
                self.train_task_pde_loss_val[task] = self.task_f_losses[task]
                self.train_task_total_loss_val[task] = self.task_losses_i[task]


                self.scaled_loss_total += self.task_losses_i[task]
        self.first_call = False
        
        return self.scaled_loss_total

    def gdro_weight_loss(self):
        q_prim_t_1 = {}
        for i, task in enumerate(self.nus):
            q_prim_t_1[task] = self.probs[task].item()*torch.exp(self.lr_q*self.loss_t_1[task].detach())
        q_prim_sum = sum(q_prim_t_1.values())
        for i, task in enumerate(self.nus):
            self.probs[task] = q_prim_t_1[task]/q_prim_sum
            
    
    def gdro_loss(self):
        if self.model_type == 'hyper-pinn':
            self.hypernet.train()
        self.dnn.train()
        self.scaled_loss_total = torch.tensor([0.0]).to(self.device)
        for i, task in enumerate(self.nus):
            self.nu = task                
            self.t_train_f = self.epoch_dataset[task]['t_train_f']
            self.x_train_f = self.epoch_dataset[task]['x_train_f']
            
            if self.model_type == 'hyper-pinn':
                net_weight, net_bias = self.hypernet(torch.tensor([self.nu]).to(self.device))
                self.dnn.set_weight(net_weight, net_bias)  
                latent = None
            elif self.model_type == 'mad-pinn':
                if self.first_call is True:
                    self.latent_id[task] = i
                latent = self.dnn.latent_vector[self.latent_id[task]]
                
            f_pred = self.net_f_2d(self.t_train_f, self.x_train_f, self.pde, latent)
            loss_f = torch.mean(f_pred ** 2)
            u_pred = self.net_u_2d(self.t_train, self.x_train, self.pde, latent)
            loss_u = torch.mean((self.u_train - u_pred) ** 2)

            
            if self.model_type == 'hyper-pinn':
                loss = (loss_u + self.f_scale * loss_f)
            elif self.model_type == 'mad-pinn':
                loss_reg = torch.mean(torch.square(self.latent_vector))
                loss = (loss_u + self.f_scale * loss_f + self.latent_reg * loss_reg) 

            else:
                NotImplementedError
            
            self.loss_t_1[task] = loss
            self.train_task_init_loss_val[task] = loss_u.detach().item()
            self.train_task_pde_loss_val[task] = loss_f.detach().item()
            self.u_loss_total += loss_u.detach().item()
            self.f_loss_total += loss_f.detach().item()
         
        if self.iter % self.prob_period == 0:     
            self.gdro_weight_loss()
        for i, task in enumerate(self.nus):
            self.prob_per_epoch[task].append(self.probs[task].item())
            scaled_loss = self.probs[task].item()* self.loss_t_1[task]
            self.scaled_loss_total += scaled_loss
            self.train_task_total_loss_val[task] = scaled_loss.detach().item()
            
        self.first_call = False
        self.scaled_loss_total.backward(retain_graph=True)
    
    
    def select_forward(self):
        return torch.sigmoid(self.selectnet[self.nu])  
    
    def self_pace_weight_loss(self):
        self.scaled_loss_total = torch.tensor([0.0]).to(self.device)

        for i, task in enumerate(self.nus):
            self.nu = task                
            self.t_train_f = self.epoch_dataset[task]['t_train_f']
            self.x_train_f = self.epoch_dataset[task]['x_train_f']
            
            self.probs[task] = self.select_forward()
            self.prob_per_epoch[task].append(self.probs[task].item())
    
            if self.model_type == 'hyper-pinn':
                net_weight, net_bias = self.hypernet(torch.tensor([self.nu]).to(self.device))
                self.dnn.set_weight(net_weight, net_bias)  
                latent = None
            elif self.model_type == 'mad-pinn':
                if self.first_call is True:
                    self.latent_id[self.nu] = i
                latent = self.dnn.latent_vector[self.latent_id[self.nu]]
                
            f_pred = self.net_f_2d(self.t_train_f, self.x_train_f, self.pde, latent)
            loss_f = torch.mean(f_pred ** 2)
            u_pred = self.net_u_2d(self.t_train, self.x_train, self.pde, latent)
            loss_u = torch.mean((self.u_train - u_pred) ** 2)
            
            if self.model_type == 'hyper-pinn':
                scaled_loss = self.probs[task] *(loss_u + self.f_scale * loss_f)
            elif self.model_type == 'mad-pinn':
                loss_reg = torch.mean(torch.square(self.latent_vector))
                scaled_loss = self.probs[task] * (loss_u + self.f_scale * loss_f + self.latent_reg * loss_reg) 

            else:
                NotImplementedError
                
            if self.iter %1000 == 0:
                 self.pace_k -= self.pace_k_step
        
            # for plotting
            self.train_task_init_loss_val[task] = loss_u.detach().item()
            self.train_task_pde_loss_val[task] = loss_f.detach().item()
            self.train_task_total_loss_val[task] = scaled_loss

            self.scaled_loss_total += scaled_loss
            self.u_loss_total += loss_u.detach().item()
            self.f_loss_total += loss_f.detach().item()
            
        self.first_call = False
        self.scaled_loss_total = self.scaled_loss_total - (1/self.pace_k) * sum(tensor.sum() for tensor in self.probs.values()) + torch.mean(sum(tensor for tensor in self.probs.values()) - 1)
        self.scaled_loss_total.backward(retain_graph=True)
        return self.scaled_loss_total    

    def self_pace_meta_loss(self):
        if self.model_type == 'hyper-pinn':
            self.hypernet.train()
        self.dnn.train()
        self.scaled_loss_total = torch.tensor([0.0]).to(self.device)
        for i, task in enumerate(self.nus):
            self.nu = task                
            self.t_train_f = self.epoch_dataset[task]['t_train_f']
            self.x_train_f = self.epoch_dataset[task]['x_train_f']
            if self.model_type == 'hyper-pinn':
                net_weight, net_bias = self.hypernet(torch.tensor([self.nu]).to(self.device))
                self.dnn.set_weight(net_weight, net_bias)  
                latent = None
            elif self.model_type == 'mad-pinn':
                if self.first_call is True:
                    self.latent_id[task] = i
                latent = self.dnn.latent_vector[self.latent_id[task]]
                
            f_pred = self.net_f_2d(self.t_train_f, self.x_train_f, self.pde, latent)
            loss_f = torch.mean(f_pred ** 2)
            u_pred = self.net_u_2d(self.t_train, self.x_train, self.pde, latent)
            loss_u = torch.mean((self.u_train - u_pred) ** 2)
            
            if self.model_type == 'hyper-pinn':
                scaled_loss = self.probs[task].item() *(loss_u + self.f_scale * loss_f)
            elif self.model_type == 'mad-pinn':
                loss_reg = torch.mean(torch.square(self.latent_vector))
                scaled_loss = self.probs[task].item()* (loss_u + self.f_scale * loss_f + self.latent_reg * loss_reg) 

            else:
                NotImplementedError

            # for plotting
            self.train_task_init_loss_val[task] = loss_u.detach().item()
            self.train_task_pde_loss_val[task] = loss_f.detach().item()
            self.train_task_total_loss_val[task] = scaled_loss

            self.scaled_loss_total += scaled_loss
            self.u_loss_total += loss_u.detach().item()
            self.f_loss_total += loss_f.detach().item()

        self.first_call = False
        self.scaled_loss_total = self.scaled_loss_total - (1/self.pace_k) * sum(tensor for tensor in self.probs.values()) 
        self.scaled_loss_total.backward()
        return self.scaled_loss_total
   
    def RAD(self):
        self.optimizer.zero_grad()
        ps = {}
        if self.model_type == 'hyper-pinn':
            net_weight, net_bias = self.hypernet(torch.tensor([self.nu]).to(self.device))
            self.dnn.set_weight(net_weight, net_bias)  
            latent = None
        elif self.model_type == 'mad-pinn':
            latent = self.dnn.latent_vector[self.latent_id[self.nu]]
        
        t_test = t_test.to(self.device)
        x_test = x_test.to(self.device)
        u_pred = self.net_u_2d(t_test, x_test, self.pde, latent)
        f_pred = Equation(u_pred, t_test, x_test, self.nu)
        p = (torch.abs(f_pred)/torch.mean(torch.abs(f_pred)) + 1) / torch.sum(torch.abs(f_pred)/torch.mean(torch.abs(f_pred)) + 1)
        ps['p'] = p
        ps['ts'] = t_test
        ps['xs'] = x_test
            
        return ps
        
    def generate_epoch_dataset(self, iteration):
    
        if self.residual_sampling is True and self.sampling_type != 'uniform' and self.iter % self.sampler['period']==0 and self.iter > 1000:
            # for task in self.selected_tasks:
            if self.hparams.residual_sampler['type'] == 'rad':
                t_test, x_test, _, _, _ = Generate_train_data(
                    20000, self.num_ic_val, self.num_bc_val, self.device)
                for task in self.nus:
                    self.nu = task
                    p = self.RAD(t_test, x_test)
                    self.ps[task] = p
            else:
                t_train_f, x_train_f, _, _, _ = Generate_train_data(
                            20000, self.num_ic, self.num_bc, self.device)
            for task in self.nus:
                num_train =  int(self.probs[task]*self.budget)
                num_train = num_train - self.task_budget[task][-1]
                if num_train>0:
                    if self.hparams.residual_sampler['type'] == 'rad':
                        _, top_indices = torch.topk(self.ps[task]['p'], k=num_train,  dim=0)
                        self.epoch_dataset[task]['t_train_f'] = torch.cat((self.epoch_dataset[task]['t_train_f'], self.ps[task]['ts'][top_indices.squeeze(dim=1)]), dim=0)
                        self.epoch_dataset[task]['x_train_f'] = torch.cat((self.epoch_dataset[task]['x_train_f'], self.ps[task]['xs'][top_indices.squeeze(dim=1)]), dim=0)
                        self.task_budget[task].append(num_train + self.task_budget[task][-1]) 
                    else:
                        self.epoch_dataset[task]['t_train_f'] = torch.cat((self.epoch_dataset[task]['t_train_f'], t_train_f[0:num_train]), dim=0)
                        self.epoch_dataset[task]['x_train_f'] = torch.cat((self.epoch_dataset[task]['x_train_f'], x_train_f[0:num_train]), dim=0)
                        self.task_budget[task].append(num_train + self.task_budget[task][-1]) 
                        
                elif num_train == 0:
                    self.task_budget[task].append(self.task_budget[task][-1]) 
                else:
                    if np.abs(num_train) >= self.epoch_dataset[task]['t_train_f'].shape[0]:
                        self.epoch_dataset[task]['t_train_f'] = self.epoch_dataset[task]['t_train_f'][0:10]
                        self.epoch_dataset[task]['x_train_f'] = self.epoch_dataset[task]['x_train_f'][0:10]
                        
                        self.task_budget[task].append(10) 
                    else:
                        self.epoch_dataset[task]['t_train_f'] = self.epoch_dataset[task]['t_train_f'][np.abs(num_train):self.epoch_dataset[task]['t_train_f'].shape[0]]
                        self.epoch_dataset[task]['x_train_f'] = self.epoch_dataset[task]['x_train_f'][np.abs(num_train):self.epoch_dataset[task]['x_train_f'].shape[0]]
                        self.task_budget[task].append(self.task_budget[task][-1] + num_train) 
                    
            for task in self.nus:
                # if task not in self.selected_tasks:
                if task not in self.nus:
                    self.task_budget[task].append(self.task_budget[task][-1])  
        
        else:
            if iteration % self.hparams.residual_sampler['period'] == 0:                             
                if self.hparams.residual_sampler['type'] == 'random':
                    self.t_train_f, self.x_train_f, _, _, _ = Generate_train_data(
                    self.num_train, self.num_ic, self.num_bc, self.device)
                    for task in self.nus:
                        self.epoch_dataset[task] = {'t_train_f': self.t_train_f, 'x_train_f': self.x_train_f}
                        self.task_budget[task].append(self.task_budget[task][-1])
                elif self.hparams.residual_sampler['type'] == 'half_random':
                    half = int(self.num_train/2)
                    t_train_f, x_train_f, _, _, _ = Generate_train_data(
                        half, self.num_ic, self.num_bc, self.device)
                    for task in self.nus:
                        self.nu = task
                        self.epoch_dataset[task]['t_train_f'][half:] =  t_train_f
                        self.epoch_dataset[task]['x_train_f'][half:] =  x_train_f
                        self.task_budget[task].append(self.num_train)        
                    
                elif self.hparams.residual_sampler['type'] == 'rad':
                    if self.hparams.residual_sampler['adaptive_type'] == 'gradual':
                        t_test, x_test, _, _, _ = Generate_train_data(
                             20000, self.num_ic_val, self.num_bc_val, self.device)
                        for task in self.nus:
                            self.nu = task
                            ps = self.RAD(t_test, x_test)
                            points_no = self.adaptive_points_no
                            _, top_indices = torch.topk(ps['p'], k=points_no,  dim=0)
                            self.epoch_dataset[task]['t_train_f'] = torch.cat((self.epoch_dataset[task]['t_train_f'], ps['ts'][top_indices.squeeze(dim=1)]), dim=0)
                            self.epoch_dataset[task]['x_train_f'] = torch.cat((self.epoch_dataset[task]['x_train_f'], ps['xs'][top_indices.squeeze(dim=1)]), dim=0)
                            self.task_budget[task].append(self.epoch_dataset[task]['t_train_f'].shape[0])  
                    else:
                        if iteration % self.hparams.residual_sampler['period'] == 0:
                            t_test, x_test, _, _, _ = Generate_train_data(
                                20000, self.num_ic_val, self.num_bc_val, self.device)
                            for task in self.nus:
                                self.nu = task
                                ps = self.RAD(t_test, x_test)
                                half = int(self.num_train/2)
                                _, top_indices = torch.topk(ps['p'], k=half,  dim=0)
                                self.epoch_dataset[task]['t_train_f'] = torch.cat((self.epoch_dataset[task]['t_train_f'][0:half], ps['ts'][top_indices.squeeze(dim=1)]), dim=0)
                                self.epoch_dataset[task]['x_train_f'] = torch.cat((self.epoch_dataset[task]['x_train_f'][0:half], ps['xs'][top_indices.squeeze(dim=1)]), dim=0)
                                self.task_budget[task].append(self.num_train)        
                else:
                    for task in self.nus:
                        self.task_budget[task].append(self.task_budget[task][-1])
            else:
                for task in self.nus:
                    self.task_budget[task].append(self.task_budget[task][-1])
                                           
    def train(self):
        train_av_init_loss = []
        train_av_pde_loss = []
        train_av_tot_loss = []
        lrs = []
                
        eval_metrics = {}
        losses = {}
        epochs = 0
        
        start = time.time()
        print("start time: ", start )
        epoch_no_loss_improve = 0
        
        org_sampling = self.sampling_type
        for it in tqdm(range(self.max_iter)):
            self.iter = it   
            if it < 200:
                self.sampling_type = 'uniform'
            else:
                self.sampling_type = org_sampling
            
            if self.model_type == 'hyper-pinn':
                self.hypernet.train()

            self.dnn.train()
            if it !=0:
                self.generate_epoch_dataset(it)   
            
            
            if self.optim == 'lbfgs':
                self.optimizer.step(self.loss_func_2d)
                self.StepLR.step()
                for task in self.nus:
                    self.train_task_init_loss[task].append(self.train_task_init_loss_val[task].item())
                    self.train_task_pde_loss[task].append(self.train_task_pde_loss_val[task].item())
                    self.train_task_total_loss[task].append(self.train_task_total_loss_val[task].item())
                    # self.prob_per_epoch[task].append(self.prob_per_epoch_val[task])
                    
                    """print('Task %0.e probability:  %.5e' % (task, self.prob_per_epoch_val[task]))
                    print('Task %0.e, Iter %d, Loss: %.5e, Loss_u: %.5e, Loss_f: %.5e' % (task, \
                        it + 1, self.train_task_total_loss_val[task], self.train_task_init_loss_val[task], self.train_task_pde_loss_val[task]))"""
                    # sys.stdout.flush()
            
            elif self.optim == 'adam':    
                if self.sampler['type'] == 'self_pace_gdro':
                    self.optimizer.zero_grad()
                    self.gdro_loss()
                    self.optimizer.step()
                elif self.sampler['type'] == 'self_pace':
                    self.select_optimizer.zero_grad()
                    self.self_pace_weight_loss()
                    self.select_optimizer.step()
                    self.optimizer.zero_grad()
                    self.self_pace_meta_loss()
                    self.optimizer.step()
                    
                else:
                    self.optimizer.zero_grad()
                    self.loss_func_2d()
                    self.optimizer.step()
                self.StepLR.step()
                lrs.append(self.optimizer.param_groups[0]['lr'])

            else:
                self.optimizer.zero_grad()
                u_pred = self.net_u_2d(self.t_train, self.x_train, self.pde)
                f_pred = self.net_f_2d(self.t_train_f, self.x_train_f, self.pde)
                loss_u = torch.mean((self.u_train - u_pred) ** 2)
                loss_f = torch.mean(f_pred ** 2)
                loss = loss_u + loss_f
                loss.backward()
                self.optimizer.step()
                self.StepLR.step()
                if it % 100 == 0:
                    print('Iter %d, Loss: %.5e, Loss_u: %.5e, Loss_f: %.5e' % (
                            it, loss.item(), loss_u.item(), loss_f.item()))
                    sys.stdout.flush()

            
            av_l2_loss = self.test(it, self.pde)
            self.iter += 1
                
            train_av_init_loss.append(self.u_loss_total.item()/len(self.nus))
            train_av_pde_loss.append(self.f_loss_total.item()/len(self.nus))
            train_av_tot_loss.append((self.u_loss_total.item() + self.f_loss_total.item())/len(self.nus))
            
            
            if av_l2_loss < self.av_l2_loss:
                self.av_l2_loss = av_l2_loss
                if self.model_type == 'hyper-pinn':
                    torch.save({
                        'epoch': epochs,
                        'model_state_dict': self.hypernet.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        }, self.output_path + '/best_model.pt')
                elif self.model_type == 'mad-pinn':
                    torch.save({
                        'epoch': epochs,
                        'model_state_dict': self.dnn.state_dict(),
                        'z_map': self.latent_id,
                        'z_i': self.dnn.latent_vector,
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        }, self.output_path + '/best_model.pt')
                
            elif np.isnan(av_l2_loss):
                print("Early stopping -- loss is NAN")
                return
            
            else:
                epoch_no_loss_improve += 1
            
            epochs += 1
            if epochs == self.max_iter:
                print("Training Finished! To evaluate run stage 2")
            if self.iter%100 == 0:
                print(lrs[-1])

            torch.cuda.empty_cache()
            del self.scaled_loss_total

            
        
        end = time.time()
        print("end time: ", end)
        print("train time: ", end - start)
        
        losses['train_task_init_loss'] = self.train_task_init_loss
        losses['train_task_pde_loss'] = self.train_task_pde_loss
        losses['train_task_total_loss'] = self.train_task_total_loss
        
        losses['train_av_init_loss'] = train_av_init_loss
        losses['train_av_pde_loss'] = train_av_pde_loss
        losses['train_av_total_loss'] = train_av_tot_loss
        losses['lr'] = lrs
        
        losses['prob_per_epoch'] = self.prob_per_epoch
        losses['gi_gsum_per_epoch'] = self.gi_gsum_per_epoch
        
        losses['l2'] = self.l2_loss
        losses['mse'] = self.mse_loss

        losses['epochs'] = epochs
            
        eval_metrics['time'] = end - start
        with open(self.output_path + '/eval.json', 'w') as handle:
            json.dump(eval_metrics, handle)
        
        total = 0
        for task in self.task_budget.keys():
            total += self.task_budget[task][-1]
        self.task_budget['total'] = total
        with open(self.output_path + '/budget.json', 'w') as handle:
            json.dump(self.task_budget, handle)
            
        with open(self.output_path + '/loss_values.json', 'w') as handle:
            json.dump(losses, handle)
                    
    def evaluate(self):
        tag = self.tag
        hparams = self.hparams
        
        if self.model_type == 'mad-pinn':
           output_path = "results_burger_mad/figures/{}".format(self.tag)
        else:
            output_path = "results_burger_hyper/figures/{}".format(self.tag)  

        with open(output_path + '/loss_values.json', 'r') as handle:
            losses = json.load(handle)
            
        prob_per_epoch = losses['prob_per_epoch'] 
        gigsum_per_epoch = losses['gi_gsum_per_epoch']
        
        train_av_init_loss = losses['train_av_init_loss'] 
        train_av_pde_loss = losses['train_av_pde_loss']
        train_av_tot_loss = losses['train_av_total_loss'] 
        
        l2_vals = losses['l2']
        epochs = len(l2_vals[list(l2_vals.keys())[0]])
        
        nus = list(l2_vals.keys())
        
        l2_loss = {}
        for task in l2_vals.keys():
            l2_loss[task] = l2_vals[task][-1]
        with open(self.output_path + '/l2_loss.json', 'w') as handle:
            json.dump(l2_loss, handle)
        
        mse_loss = losses['mse']
    
        selected_nus = nus
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.nus)))
        utp.plot_val_over_epoch(output_path, 'prob_epoch.png', epochs, selected_nus, prob_per_epoch, colors, 'Epochs', 'Probability values','Probability values for each task across epochs')
        utp.plot_val_over_epoch(output_path, 'gigsum_epoch.png', epochs, selected_nus, gigsum_per_epoch, colors, 'Epochs', 'gigsum values','gigsum values for each task across epochs')
        
        utp.plot_val_over_epoch(output_path, 'test_l2.png', epochs, selected_nus, l2_vals, colors, 'Epochs', 'loss values','l2 loss values for each task across epochs')

        utp.plot_val_over_epoch(output_path, 'test_mse.png', epochs, selected_nus, mse_loss, colors, 'Epochs', 'loss values','mse loss values for each task across epochs')
        
        utp.calculate_disparity(output_path, l2_vals, 1, 'l2')
        utp.calculate_disparity(output_path, l2_vals, 5, 'l2')
        utp.calculate_disparity(output_path, l2_vals, 10, 'l2')
        
        utp.calculate_disparity(output_path, mse_loss, 1, 'mse')
        utp.calculate_disparity(output_path, mse_loss, 5, 'mse')
        utp.calculate_disparity(output_path, mse_loss, 10, 'mse')
        
        utp.plot_loss_convergence(output_path, selected_nus, colors, "mse_conv", mse_loss, hparams.loss['mse_treshold'])
        utp.plot_loss_convergence(output_path, selected_nus, colors, "l2_conv", l2_vals, hparams.loss['l2_treshold'])

        
    def test(self, it, pde):
        if self.model_type =='hyper-pinn':
            self.hypernet.eval()
        if self.model_type =='mad-pinn':
            self.dnn.eval()
        else:
            NotImplementedError
        av_l2_loss = 0
        l2_loss_min = 100
        l2_loss_max = -100
        for i, nu in enumerate(self.nus):
            self.nu = nu
            # net_weight, net_bias = self.hypernet(torch.tensor([self.nu]))
            if self.model_type == 'hyper-pinn':
                net_weight, net_bias = self.hypernet(torch.tensor([self.nu]).to(self.device))
                self.dnn.set_weight(net_weight, net_bias)  
                latent = None
            elif self.model_type == 'mad-pinn':
                latent = self.dnn.latent_vector[self.latent_id[nu]]
            
            if it % 10 == 0 or it== self.max_iter or it==self.max_iter-1:
                lambda_1 = None
                self.x_inverse_data, self.t_inverse_data, self.u_inverse_data, self.u_test, self.t_test, self.x_test = Load_ground_truth(self.nu)
                l2_loss, mse_loss = pde_test(pde, self.nu, self.t_test, self.x_test, self.u_test, lambda_1, self.net_u_2d, self.problem, it, self.loss_list, self.output_path, self.tag+str(self.nu), self.max_iter - 1, latent, self.device)
                self.l2_loss[self.nus[i]].append(l2_loss)
                self.mse_loss[self.nus[i]].append(mse_loss)
                av_l2_loss += l2_loss 
            else:
                lambda_1 = None
                self.l2_loss[self.nus[i]].append(self.l2_loss[self.nus[i]][-1])
                self.mse_loss[self.nus[i]].append(self.mse_loss[self.nus[i]][-1])
                av_l2_loss += self.l2_loss[self.nus[i]][-1]
                
            
        return av_l2_loss
    
    def fine_tune(self, task):
        l2_losses = []
        mse_losses = []
        self.iter = 100
        checkpoint = torch.load(self.output_path + '/best_model.pt')
        self.dnn.load_state_dict(checkpoint['model_state_dict'])
        self.dnn = self.dnn.to(self.device)
        z_map = checkpoint['z_map']        
        self.output_path = self.output_path + '/test_time/'
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        if not os.path.exists(self.output_path + '/epoch_figs'):
            os.makedirs(self.output_path + '/epoch_figs')
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        lambda_1 = None  
        self.dnn.load_state_dict(checkpoint['model_state_dict'])
        self.nu = task
        latent_init, task_closest_to_nu = utp.find_closest(z_map, self.dnn.latent_vector, task)
        for param in self.dnn.net.parameters():
            param.requires_grad = False
        self.dnn.latent_vector = torch.nn.Parameter(latent_init.view(-1,1).clone().to(self.device), requires_grad=True)
        self.optimizer = optim.Adam(
            [self.dnn.latent_vector],
            lr=0.01,
        )  
        self.StepLR = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[2500], gamma=0.1)
        self.optimizer.zero_grad()
        iter_no_improve = 0
        total_loss = 1000
        for iter in range(self.fine_tune_iter):            
                def closure():
                    self.optimizer.zero_grad()
                    with torch.enable_grad():
                        f_pred = self.net_f_2d(self.t_train_f, self.x_train_f, self.pde, self.dnn.latent_vector)
                        loss_f = torch.mean(f_pred ** 2)
                        u_pred = self.net_u_2d(self.t_train, self.x_train, self.pde, self.dnn.latent_vector)
                        loss_u = torch.mean((self.u_train - u_pred) ** 2)
                        loss_reg = torch.mean(torch.square(self.latent_vector))
                        scaled_loss = loss_u + self.f_scale * loss_f

                    scaled_loss.backward()
                    return scaled_loss
                self.optimizer.step(closure)
                self.StepLR.step()
                self.x_inverse_data, self.t_inverse_data, self.u_inverse_data, self.u_test, self.t_test, self.x_test = Load_ground_truth(self.nu)
                l2_loss, mse_loss = pde_test(self.pde, self.nu, self.t_test, self.x_test, self.u_test, lambda_1, self.net_u_2d, self.problem, iter, self.loss_list, self.output_path, self.tag+str(self.nu), self.max_iter - 1, self.dnn.latent_vector, self.device)
                l2_losses.append(l2_loss)
                mse_losses.append(mse_loss)
                
                if l2_loss < total_loss:
                    total_loss = l2_loss
                    iter_no_improve = 0
                else:
                    iter_no_improve += 1
                
                if iter_no_improve>=400:
                    final_iter = iter
                    break
                else:
                    final_iter = self.fine_tune_iter
        
        return l2_losses, mse_losses, final_iter
   

    def generalization(self, device):
        checkpoint = torch.load(self.output_path + '/best_model.pt')
        
        if self.model_type == 'hyper-pinn':
            self.hypernet.load_state_dict(checkpoint['model_state_dict'], strict = False)
            self.hypernet.eval()
        elif self.model_type == 'mad-pinn':
            self.dnn.eval()
        losses = {}
        self.iter = 100
        self.output_path = self.output_path + '/test_time/'
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        if not os.path.exists(self.output_path + '/epoch_figs'):
            os.makedirs(self.output_path + '/epoch_figs')
        l2_losses = {task: [] for task in self.test_nus}
        mse_losses = {task: [] for task in self.test_nus}
        for i, nu in enumerate(self.test_nus):
            self.nu = nu
            self.x_inverse_data, self.t_inverse_data, self.u_inverse_data, self.u_test, self.t_test, self.x_test = Load_ground_truth(self.nu)            
            if self.model_type == 'hyper-pinn':
                net_weight, net_bias = self.hypernet(torch.tensor([self.nu]).to(device))
                self.dnn.set_weight(net_weight, net_bias)  
                latent = None
            elif self.model_type == 'mad-pinn':
                latent = self.dnn.latent_vector[self.latent_id[nu]]
        
            lambda_1 = None
            l2_loss, mse_loss = pde_test(self.pde, self.nu, self.t_test, self.x_test, self.u_test, lambda_1, self.net_u_2d, self.problem, 1000, self.loss_list, self.output_path, self.tag+str(self.nu), self.max_iter - 1, latent, device)
            l2_losses[nu].append(l2_loss)
            mse_losses[nu].append(mse_loss)
            losses[nu] = {"l2": l2_loss.item(), "mse": mse_loss.item()}
            
        with open(self.output_path + '/eval.json', 'w') as handle:
            json.dump(losses, handle)
            
        utp.calculate_disparity(self.output_path , l2_losses, 1, 'l2')
