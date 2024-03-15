import os
import json
import time
import math
import torch
import torch.nn as nn
import shutil
import random
import numpy as np
from tqdm import tqdm
import utils.plot as utp
import utils.lr as utr
from matplotlib import rc
rc('text', usetex=False)
from Helmholtz_2d import Equation, Test, Load_ground_truth, Generate_train_data, Generate_collocation_points, helmholtz_2d_source_term, helmholtz_2d_exact_u


class PINN():
    def __init__(self, network, device, model_type, hypernet, hparams, json_path):
        
        self.hparams = hparams
        self.model_type = model_type
        self.tag =self.hparams.pde + '_' + model_type + '_' + hparams.sampler['type'] + '_' + str(hparams.id) + '_' + str(hparams.seed) + '_num_' + str(hparams.train_set['num_train'])
        # device
        self.device = device
        # deep neural networks
        self.network = network
        self.hypernet = hypernet
        if self.model_type == 'mad-pinn':
            self.latent_vector = self.network.latent_vector
            self.latent_id = {}
            self.latent_reg = hparams.network['latent_reg']
        else:
            self.latent_vector = None
            self.latent_reg = None
        # number of train&val points
        self.num_train = hparams.train_set['num_train']
        self.num_bc = hparams.train_set['num_bc']
        self.num_val_res = hparams.validate_set['num_res']
        self.num_bc_val = hparams.validate_set['num_bc']
        self.f_scale = hparams.train_set['f_scale']
        self.num_test = hparams.train_set['num_test']
        # optimizer and scheduler

        self.optim = 'adam'
        if self.optim == 'lbfgs':
            self.optimizer = torch.optim.LBFGS(self.hypernet.parameters(), lr=0.1, line_search_fn="strong_wolfe")
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.5)
        elif self.optim == 'adam':
            if self.model_type == 'hyper-pinn':
                self.optimizer = torch.optim.Adam(self.hypernet.parameters(), lr=1e-3)
            elif self.model_type == 'mad-pinn':
                self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3)
        self.StepLR = utr.CosineAnnealingLR(self.optimizer, T_max=20000, warmup_steps=300)
        # PDE parameter setting
        self.params = []
        for i in range(0, 3):
            for j in range(0, 3):
                self.params.append((round(0.5 * i + 0.5, 2), round(0.5 * j + 0.5, 2)))
        self.coefficient = 1.0
        self.ic_func = lambda x: np.exp(-np.power((x - np.pi) / (np.pi / 4), 2.) / 2.)
        # task sampler setting
        self.sampler = hparams.sampler
        self.prob_period = self.sampler['period']
        random.shuffle(self.params)
        self.probs = {}
        for task in self.params:
            self.probs[task] = torch.ones((1, 1)) * (1 / len(self.params))
        self.lr_q = self.sampler['eta']
        self.sampling_type = hparams.sampler['type']
        self.config_sampling = hparams.sampler['type']
        self.residual_sampling = hparams.sampler['residual_sampling']
        # self.budget = hparams.sampler['task_max_budget'] * len(self.params)
        self.budget = self.num_train * len(self.params)
        self.kl_type = hparams.sampler['kl']
        self.epsilon = hparams.sampler['epsilon']
        self.prob_period = hparams.sampler['period']
        self.eta = hparams.sampler['eta']
        self.start_random = hparams.sampler['start_random']
        # for plotting
        self.train_task_init_loss = {task: [] for task in self.params}
        self.train_task_pde_loss = {task: [] for task in self.params}
        self.train_task_total_loss = {task: [] for task in self.params}
        self.l2_loss = {task: [] for task in self.params}
        self.mse_loss = {task: [] for task in self.params}
        self.prob_per_epoch = {task: [] for task in self.params}
        self.gi_gsum_per_epoch = {task: [] for task in self.params}
        self.train_task_init_loss_val = {}
        self.train_task_pde_loss_val = {}
        self.train_task_total_loss_val = {}
        self.epoch_dataset = {}
        self.epoch_dataset_val = {}
        self.scaled_loss_total = torch.tensor([0.0]).to(self.device)
        self.u_loss_total = torch.tensor([0.0]).to(self.device)
        self.f_loss_total = torch.tensor([0.0]).to(self.device)
        self.av_l2_loss = 10000
        # output path
        self.tag = str(hparams.id) + '_' + str(hparams.seed)
        self.output_path = "results/figures/{}".format(self.tag)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        if not os.path.exists(self.output_path + '/epoch_figs'):
            os.makedirs(self.output_path + '/epoch_figs')
        shutil.copy(json_path, self.output_path)
        # train
        self.max_iter = hparams.max_iter
        self.iter = 0
        self.random_f = False
        # test
        self.test_params = hparams.test_params
        random.shuffle(self.test_params)
        self.task_budget = {task: [self.num_train] for task in self.params}
        
        self.t_train_f, self.x_train_f = Generate_collocation_points(self.num_train, device)
        self.t_train, self.x_train = Generate_train_data(self.num_bc, self.device)
        self.t_val_f, self.x_val_f = Generate_collocation_points(self.hparams.validate_set['num_res'], device)
        self.t_val, self.x_val = Generate_train_data(self.hparams.validate_set['num_bc'], self.device)
        for i, task in enumerate(self.params):
            a1, a2 = task
            self.a1, self.a2 = torch.tensor([a1]), torch.tensor([a2])
            u_s = helmholtz_2d_source_term(self.t_train_f, self.x_train_f, a1, a2, self.coefficient)
            u_b = helmholtz_2d_exact_u(self.t_train, self.x_train, a1, a2)
            self.epoch_dataset[task] = {'t_train_f': self.t_train_f, 'x_train_f': self.x_train_f, 'u_train_f': u_s, 'u_train_b': u_b}
            
            u_s_v = helmholtz_2d_source_term(self.t_val_f, self.x_val_f, a1, a2, self.coefficient)
            u_b_v = helmholtz_2d_exact_u(self.t_val, self.x_val, a1, a2)
            self.epoch_dataset_val[task] = {'t_val_f': self.t_val_f, 'x_val_f': self.x_val_f, 'u_val_f': u_s_v, 'u_val_b': u_b_v}
            
            self.task_budget[task].append(self.t_train_f.shape[0])
        
        self.loss_t_1 = {}
        self.selectnet = {}
        for i, task in enumerate(self.params):
            self.selectnet[task] = nn.Parameter(torch.tensor(1.0, requires_grad=True).float().to(self.device))
        self.select_optimizer = torch.optim.Adam(self.selectnet.values(), lr=1e-3)
        
        self.ps = {}
                
        if (self.residual_sampling == False or self.sampling_type == 'uniform') and self.hparams.residual_sampler['type'] == 'rad' and self.hparams.residual_sampler['adaptive_type'] == 'gradual':
            steps = (self.max_iter / self.hparams.residual_sampler['period'] - 2)
            self.adaptive_points_no = math.floor(int(self.num_train/2) / steps)
            self.num_train = int(self.num_train/2) + round(steps * (int(self.num_train/2) / steps - self.adaptive_points_no))
            self.t_train_f, self.x_train_f = Generate_collocation_points(self.num_train, device)
            for i, task in enumerate(self.params):
                a1, a2 = task
                self.a1, self.a2 = torch.tensor([a1]), torch.tensor([a2])
                u_s = helmholtz_2d_source_term(self.t_train_f, self.x_train_f, a1, a2, self.coefficient)
                self.epoch_dataset[task]['t_train_f'] = self.t_train_f
                self.epoch_dataset[task]['x_train_f'] = self.x_train_f
                self.epoch_dataset[task]['u_train_f'] = u_s
        # self.StepLR = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.8)
        self.first_call = True
                

    def task_sampler(self):
        gradients_i = {}
        gradients_j = {}
        self.gradient_training = {}
        task_losses_i = {}
        task_losses_j = {}
        probs = {}
        task_u_losses = {}
        task_f_losses = {}
      
        for i, task in enumerate(self.params):
            self.optimizer.zero_grad()
            self.t_train_f = self.epoch_dataset[task]['t_train_f'].to(self.device)
            self.x_train_f = self.epoch_dataset[task]['x_train_f'].to(self.device)
            self.u_train_f = self.epoch_dataset[task]['u_train_f'].to(self.device)
            self.u_b = self.epoch_dataset[task]['u_train_b'].to(self.device)
            
            self.t_val_f = self.epoch_dataset_val[task]['t_val_f'].to(self.device)
            self.x_val_f = self.epoch_dataset_val[task]['x_val_f'].to(self.device)
            self.u_val_f = self.epoch_dataset_val[task]['u_val_f'].to(self.device)
            self.u_b_v = self.epoch_dataset_val[task]['u_val_b'].to(self.device)
            
            
            a1, a2 = task
            self.a1, self.a2 = torch.tensor([a1]).to(self.device), torch.tensor([a2]).to(self.device)
            if self.model_type == 'hyper-pinn':
                net_weight, net_bias = self.hypernet(self.a1.to(self.device), self.a2.to(self.device))
                self.network.set_params(net_weight, net_bias)
                latent = None
            elif self.model_type == 'mad-pinn':
                latent = self.network.latent_vector[self.latent_id[task]]
                
            f_pred = self.net_f(self.t_train_f, self.x_train_f, latent)
            loss_f = torch.mean((f_pred - self.u_train_f) ** 2)
            u_pred = self.net_u(self.t_train, self.x_train, latent)
            loss_u = torch.mean((self.u_b - u_pred) ** 2)
            
            if self.model_type == 'hyper-pinn':
                scaled_loss = loss_u + self.f_scale * loss_f
                scaled_loss.backward(retain_graph=True)
                gradient_ = [param.grad.detach().clone() for param in self.hypernet.parameters()]
                self.gradient_training[task] = gradient_
            elif self.model_type == 'mad-pinn':
                loss_reg = torch.mean(torch.square(self.latent_vector))
                scaled_loss = loss_u + self.f_scale * loss_f + self.latent_reg * loss_reg
                scaled_loss.backward(retain_graph=True)
                gradient_ = [param.grad.detach().clone() for param in self.network.parameters()]
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
                    net_weight, net_bias = self.hypernet(self.a1.to(self.device), self.a2.to(self.device))
                    self.network.set_params(net_weight, net_bias)
                    latent = None
                elif self.model_type == 'mad-pinn':
                    latent = self.network.latent_vector[self.latent_id[task]]
                f_pred = self.net_f(self.t_val_f, self.x_val_f, latent)
                loss_f = torch.mean((f_pred - self.u_val_f) ** 2)
                u_pred = self.net_u(self.t_val, self.x_val, latent)
                loss_u = torch.mean((self.u_b_v - u_pred) ** 2)
                
                if self.model_type == 'hyper-pinn':
                    scaled_loss = loss_u + self.f_scale * loss_f
                    gradient_ = torch.autograd.grad(scaled_loss, self.hypernet.parameters(), retain_graph=False, allow_unused=True)
                elif self.model_type == 'mad-pinn':
                    loss_reg = torch.mean(torch.square(self.latent_vector))
                    scaled_loss = loss_u + self.f_scale * loss_f + self.latent_reg * loss_reg
                    gradient_ = torch.autograd.grad(scaled_loss, self.network.parameters(), retain_graph=False, allow_unused=True)
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
            for i, task in enumerate(self.params):
                a1, a2 = task
                self.a1, self.a2 = torch.tensor([a1]), torch.tensor([a2])
                if self.kl_type == 'uniform':
                    gigsum = torch.sum(torch.stack([torch.sqrt(task_losses_i[task] * task_losses_j[task2]) * torch.nn.functional.cosine_similarity(
                        gradients_i[task], gradients_j[task2], dim=0) for j, task2 in enumerate(self.params)]))
                    self.gi_gsum_per_epoch[task].append(gigsum.item())
                    # if self.it in self.milestones:
                    #     probs[task] = torch.ones((1, 1))/len(self.nus)
                    # else:
                    probs[task] = 1/len(self.params) * torch.exp(self.eta * gigsum)
                elif self.kl_type == 'consecutive':
                    gigsum = torch.sum(torch.stack([torch.sqrt(task_losses_i[task] * task_losses_j[task2]) * torch.nn.functional.cosine_similarity(
                            gradients_i[task], gradients_j[task2], dim=0) for j, task2 in enumerate(self.params)]))
                    self.gi_gsum_per_epoch[task].append(gigsum.item())
                    # if self.it in self.milestones:
                    #     probs[task] = torch.ones((1, 1))/len(self.nus)
                    # else:
                    probs[task] = self.probs[task].to(self.device) * torch.exp(self.eta * gigsum)
                else:
                    print("KL option is not implemented")

        prob_sum = sum(torch.sum(p) for p in probs.values() if isinstance(p, torch.Tensor))

        for task in probs.keys():
            probs[task] = probs[task] / prob_sum

        keys = list(probs.keys())
        random.shuffle(keys)
        for k in keys:
            self.probs[k] = probs[k]
            
        self.gradients_i = gradients_i
        self.task_losses_i = task_losses_i
        self.task_f_losses = task_f_losses
        self.task_u_losses = task_u_losses
            

    def net_u(self, t, x, latent):
        x = torch.cat([t, x], dim=1)
        out = self.network(x, latent)
        return out

    def net_f(self, t, x, latent):
        u = self.net_u(t, x, latent)
        f = Equation(u, t, x, self.coefficient)
        return f

    def loss_func(self):
        self.optimizer.zero_grad()
        self.scaled_loss_total = torch.tensor([0.0]).to(self.device)
        self.u_loss_total = torch.tensor([0.0]).to(self.device)
        self.f_loss_total = torch.tensor([0.0]).to(self.device)
        scaled_loss = 0
        if self.sampling_type == "uniform":
            for i, task in enumerate(self.params):
                self.t_train_f = self.epoch_dataset[task]['t_train_f'].to(self.device)
                self.x_train_f = self.epoch_dataset[task]['x_train_f'].to(self.device)
                self.u_train_f = self.epoch_dataset[task]['u_train_f'].to(self.device)
                self.u_b = self.epoch_dataset[task]['u_train_b'].to(self.device)
                a1, a2 = task
                self.a1, self.a2 = torch.tensor([a1]).to(self.device), torch.tensor([a2]).to(self.device)
                if self.model_type == 'hyper-pinn':
                    net_weight, net_bias = self.hypernet(self.a1.to(self.device), self.a2.to(self.device))
                    self.network.set_params(net_weight, net_bias)
                    latent = None
                elif self.model_type == 'mad-pinn':
                    if self.first_call is True:
                        self.latent_id[task] = i
                    latent = self.network.latent_vector[self.latent_id[task]]
                f_pred = self.net_f(self.t_train_f, self.x_train_f, latent)
                loss_f = torch.mean((f_pred - self.u_train_f) ** 2)
                u_pred = self.net_u(self.t_train, self.x_train, latent)
                loss_u = torch.mean((self.u_b - u_pred) ** 2)
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
                self.prob_per_epoch[task].append(self.probs[task])
                self.scaled_loss_total += scaled_loss
                self.u_loss_total += loss_u.detach().item()
                self.f_loss_total += loss_f.detach().item()
            self.scaled_loss_total.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.hypernet.parameters(), max_norm=10, norm_type=2)
        else:
            for i, task in enumerate(self.params):
                if self.model_type == 'mad-pinn':
                    if self.first_call is True:
                        self.latent_id[task] = i
            self.first_call = False
            self.task_sampler()            
            self.optimizer.zero_grad()
            for i, task in enumerate(self.params):
                self.t_train_f = self.epoch_dataset[task]['t_train_f'].to(self.device)
                self.x_train_f = self.epoch_dataset[task]['x_train_f'].to(self.device)
                self.u_train_f = self.epoch_dataset[task]['u_train_f'].to(self.device)
                self.u_b = self.epoch_dataset[task]['u_train_b'].to(self.device)
                a1, a2 = task
                self.a1, self.a2 = torch.tensor([a1]), torch.tensor([a2])
                if self.residual_sampling == False:
                    if self.model_type == 'hyper-pinn':
                        for p, g in zip(self.hypernet.parameters(), self.gradient_training[task]):
                            p.grad += self.probs[task].item() * g
                    elif self.model_type == 'mad-pinn':
                        for p, g in zip(self.network.parameters(), self.gradient_training[task]):
                            p.grad += self.probs[task].item() * g
                    else:
                        NotImplementedError
                else:
                    if self.model_type == 'hyper-pinn':
                        for p, g in zip(self.hypernet.parameters(), self.gradient_training[task]):
                            p.grad += g
                            
                    elif self.model_type == 'mad-pinn':
                        for p, g in zip(self.network.parameters(), self.gradient_training[task]):
                            p.grad += g
                        continue
                    else:
                        NotImplementedError

                self.train_task_init_loss_val[task] = self.task_u_losses[task]
                self.train_task_pde_loss_val[task] = self.task_f_losses[task]
                self.train_task_total_loss_val[task] = self.task_losses_i[task]
                self.prob_per_epoch[task].append(self.probs[task])


                self.scaled_loss_total += self.task_losses_i[task]
            
        self.first_call = False
        # torch.nn.utils.clip_grad_norm_(self.hypernet.parameters(), max_norm=10, norm_type=2)
        return self.scaled_loss_total


    def weight_loss(self):
        q_prim_t_1 = {}
        for i, task in enumerate(self.params):
            q_prim_t_1[task] = self.probs[task].item()*torch.exp(self.lr_q*self.loss_t_1[task].detach())
        q_prim_sum = sum(q_prim_t_1.values())
        for i, task in enumerate(self.params):
            self.probs[task] = q_prim_t_1[task]/q_prim_sum
            
    
    def self_pace_loss(self):
        if self.model_type == 'hyper-pinn':
            self.hypernet.train()
        self.network.train()
        self.scaled_loss_total = torch.tensor([0.0]).to(self.device)
        for i, task in enumerate(self.params):
            self.t_train_f = self.epoch_dataset[task]['t_train_f'].to(self.device)
            self.x_train_f = self.epoch_dataset[task]['x_train_f'].to(self.device)
            self.u_train_f = self.epoch_dataset[task]['u_train_f'].to(self.device)
            self.u_b = self.epoch_dataset[task]['u_train_b'].to(self.device)
            a1, a2 = task
            self.a1, self.a2 = torch.tensor([a1]).to(self.device), torch.tensor([a2]).to(self.device)
            if self.model_type == 'hyper-pinn':
                net_weight, net_bias = self.hypernet(self.a1.to(self.device), self.a2.to(self.device))
                self.network.set_params(net_weight, net_bias)
                latent = None
            elif self.model_type == 'mad-pinn': #to be implemented
                if self.first_call is True:
                    self.latent_id[task] = i
                latent = self.network.latent_vector[self.latent_id[task]]
            
            f_pred = self.net_f(self.t_train_f, self.x_train_f, latent)
            loss_f = torch.mean((f_pred - self.u_train_f) ** 2)
            u_pred = self.net_u(self.t_train, self.x_train, latent)
            loss_u = torch.mean((self.u_b - u_pred) ** 2)
         
            if self.model_type == 'hyper-pinn':
                loss = (loss_u + self.f_scale * loss_f)
            elif self.model_type == 'mad-pinn':
                loss_reg = torch.mean(torch.square(self.latent_vector))
                loss = (loss_u + self.f_scale * loss_f + self.latent_reg * loss_reg)
            else:
                NotImplementedError
            self.loss_t_1[task] = loss
            # for plotting
            self.train_task_init_loss_val[task] = loss_u.detach().item()
            self.train_task_pde_loss_val[task] = loss_f.detach().item()
            self.u_loss_total += loss_u.detach().item()
            self.f_loss_total += loss_f.detach().item()
        if self.iter % self.prob_period == 0:     
            self.weight_loss()
        for i, task in enumerate(self.params):
            self.prob_per_epoch[task].append(self.probs[task].item())
            scaled_loss = self.probs[task].item()* self.loss_t_1[task]
            self.scaled_loss_total += scaled_loss
            self.train_task_total_loss_val[task] = scaled_loss.detach().item()
        self.first_call = False
        self.scaled_loss_total.backward(retain_graph=True)

    def RAD(self, task):
        self.optimizer.zero_grad()
        ps = {}
        t_test, x_test = Generate_collocation_points(20000, self.device)
        u_s = helmholtz_2d_source_term(t_test, x_test, self.a1, self.a2, self.coefficient)
        if self.model_type == 'hyper-pinn':
            net_weight, net_bias = self.hypernet(self.a1.to(self.device), self.a2.to(self.device))
            self.network.set_params(net_weight, net_bias)
            latent = None
        elif self.model_type == 'mad-pinn':
            latent = self.network.latent_vector[self.latent_id[task]]
        t_test = t_test.to(self.device)
        x_test = x_test.to(self.device)
        u_pred = self.net_u(t_test, x_test, latent)
        f_pred = Equation(u_pred, t_test, x_test, self.coefficient)
        loss_f = f_pred - u_s
        p = (torch.abs(loss_f)/torch.mean(torch.abs(loss_f)) + 1) / torch.sum(torch.abs(loss_f)/torch.mean(torch.abs(loss_f)) + 1)
        ps['p'] = p
        ps['ts'] = t_test
        ps['xs'] = x_test
        ps['us'] = u_s
            
        return ps
        
    def generate_epoch_dataset(self, iteration):
        if self.residual_sampling is True and self.sampling_type != 'uniform' and self.iter % self.sampler['period']==0:
            if self.hparams.residual_sampler['type'] == 'rad':
                for task in self.params:
                    a1, a2 = task
                    self.a1, self.a2 = torch.tensor([a1]).to(self.device), torch.tensor([a2]).to(self.device)
                    p = self.RAD(task)
                    self.ps[task] = p
            for task in self.params:
                a1, a2 = task
                a1, a2 = torch.tensor([a1]).to(self.device), torch.tensor([a2]).to(self.device)
                num_train = int(self.probs[task].item()*self.budget)
                num_train = num_train - self.task_budget[task][-1]
                if num_train > 0:
                    # t_train_f, x_train_f, _, _, _ = Generate_train_data(
                    #     num_train, self.num_ic, self.num_bc, self.device)
                    if self.hparams.residual_sampler['type'] == 'rad':
                        _, top_indices = torch.topk(self.ps[task]['p'], k=num_train,  dim=0)
                        self.epoch_dataset[task]['t_train_f'] = torch.cat((self.epoch_dataset[task]['t_train_f'], self.ps[task]['ts'][top_indices.squeeze(dim=1)]), dim=0)
                        self.epoch_dataset[task]['x_train_f'] = torch.cat((self.epoch_dataset[task]['x_train_f'], self.ps[task]['xs'][top_indices.squeeze(dim=1)]), dim=0)
                        self.epoch_dataset[task]['u_train_f'] = torch.cat((self.epoch_dataset[task]['u_train_f'], self.ps[task]['us'][top_indices.squeeze(dim=1)]), dim=0)
                        self.task_budget[task].append(num_train + self.task_budget[task][-1]) 
                    else:
                        t_train_f, x_train_f = Generate_collocation_points(self.num_train, self.device)
                        u_s = helmholtz_2d_source_term(t_train_f, x_train_f, a1, a2, self.coefficient)
                        self.epoch_dataset[task]['t_train_f'] = torch.cat((self.epoch_dataset[task]['t_train_f'], t_train_f), dim=0)
                        self.epoch_dataset[task]['x_train_f'] = torch.cat((self.epoch_dataset[task]['x_train_f'], x_train_f), dim=0)
                        self.epoch_dataset[task]['u_train_f'] = torch.cat((self.epoch_dataset[task]['u_train_f'], u_s), dim=0)
                        self.task_budget[task].append(num_train + self.task_budget[task][-1]) 
                        
                elif num_train == 0:
                    self.task_budget[task].append(self.task_budget[task][-1]) 
                else:
                    if np.abs(num_train) >= self.epoch_dataset[task]['t_train_f'].shape[0]:
                        self.epoch_dataset[task]['t_train_f'] = self.epoch_dataset[task]['t_train_f'][0:10]
                        self.epoch_dataset[task]['x_train_f'] = self.epoch_dataset[task]['x_train_f'][0:10]
                        self.epoch_dataset[task]['u_train_f'] = self.epoch_dataset[task]['u_train_f'][0:10]
                        
                        self.task_budget[task].append(10) 
                    else:
                        self.epoch_dataset[task]['t_train_f'] = self.epoch_dataset[task]['t_train_f'][np.abs(num_train):self.epoch_dataset[task]['t_train_f'].shape[0]]
                        self.epoch_dataset[task]['x_train_f'] = self.epoch_dataset[task]['x_train_f'][np.abs(num_train):self.epoch_dataset[task]['x_train_f'].shape[0]]
                        self.epoch_dataset[task]['u_train_f'] = self.epoch_dataset[task]['u_train_f'][np.abs(num_train):self.epoch_dataset[task]['u_train_f'].shape[0]]                        
                        self.task_budget[task].append(self.task_budget[task][-1] + num_train) 
                                        
                                        
        else:
            if iteration % self.hparams.residual_sampler['period'] == 0: 
                if self.hparams.residual_sampler['type'] == 'random':
                    t_train_f, x_train_f = Generate_collocation_points(self.num_train, self.device)
                    for task in self.params:
                        a1, a2 = task
                        a1, a2 = torch.tensor([a1]).to(self.device), torch.tensor([a2]).to(self.device)
                        u_s = helmholtz_2d_source_term(t_train_f, x_train_f, a1, a2, self.coefficient)
                        self.epoch_dataset[task]['t_train_f'] = t_train_f
                        self.epoch_dataset[task]['x_train_f'] = x_train_f
                        self.epoch_dataset[task]['u_train_f'] = u_s
                        self.task_budget[task].append(self.task_budget[task][-1])
                elif self.hparams.residual_sampler['type'] == 'half_random':
                    half = int(self.num_train/2)
                    t_train_f, x_train_f = Generate_collocation_points(half, self.device)
                    for task in self.params:
                        a1, a2 = task
                        a1, a2 = torch.tensor([a1]).to(self.device), torch.tensor([a2]).to(self.device)
                        u_s = helmholtz_2d_source_term(t_train_f, x_train_f, a1, a2, self.coefficient)
                        self.epoch_dataset[task]['t_train_f'] = torch.cat((self.epoch_dataset[task]['t_train_f'][0:half], t_train_f), dim=0)
                        self.epoch_dataset[task]['x_train_f'] = torch.cat((self.epoch_dataset[task]['x_train_f'][0:half], x_train_f), dim=0)
                        self.epoch_dataset[task]['u_train_f'] = torch.cat((self.epoch_dataset[task]['u_train_f'][0:half], u_s), dim=0)
                        # self.epoch_dataset[task] = {'t_train_f': ps['ts'][top_indices.squeeze(dim=1)], 'x_train_f':  ps['xs'][top_indices.squeeze(dim=1)]}
                        self.task_budget[task].append(self.num_train)        
                    
                elif self.hparams.residual_sampler['type'] == 'rad':
                    if self.hparams.residual_sampler['adaptive_type'] == 'gradual':
                        for task in self.params:
                            a1, a2 = task
                            self.a1, self.a2 = torch.tensor([a1]).to(self.device), torch.tensor([a2]).to(self.device)
                            p = self.RAD(task)
                            points_no = self.adaptive_points_no
                            _, top_indices = torch.topk(p['p'], k=points_no,  dim=0)
                            self.epoch_dataset[task]['t_train_f'] = torch.cat((self.epoch_dataset[task]['t_train_f'], p['ts'][top_indices.squeeze(dim=1)]), dim=0)
                            self.epoch_dataset[task]['x_train_f'] = torch.cat((self.epoch_dataset[task]['x_train_f'], p['xs'][top_indices.squeeze(dim=1)]), dim=0)
                            self.epoch_dataset[task]['u_train_f'] = torch.cat((self.epoch_dataset[task]['u_train_f'], p['us'][top_indices.squeeze(dim=1)]), dim=0)
                            # self.epoch_dataset[task] = {'t_train_f': ps['ts'][top_indices.squeeze(dim=1)], 'x_train_f':  ps['xs'][top_indices.squeeze(dim=1)]}
                            self.task_budget[task].append(self.epoch_dataset[task]['t_train_f'].shape[0])  
                    else:
                        if iteration % self.hparams.residual_sampler['period'] == 0:
                            for task in self.params:
                                a1, a2 = task
                                self.a1, self.a2 = torch.tensor([a1]).to(self.device), torch.tensor([a2]).to(self.device)
                                p = self.RAD(task)
                                half = int(self.num_train/2)
                                _, top_indices = torch.topk(p['p'], k=half,  dim=0)
                                self.epoch_dataset[task]['t_train_f'] = torch.cat((self.epoch_dataset[task]['t_train_f'][0:half], p['ts'][top_indices.squeeze(dim=1)]), dim=0)
                                self.epoch_dataset[task]['x_train_f'] = torch.cat((self.epoch_dataset[task]['x_train_f'][0:half], p['xs'][top_indices.squeeze(dim=1)]), dim=0)
                                self.epoch_dataset[task]['u_train_f'] = torch.cat((self.epoch_dataset[task]['u_train_f'][0:half], p['us'][top_indices.squeeze(dim=1)]), dim=0)
                                # self.epoch_dataset[task] = {'t_train_f': ps['ts'][top_indices.squeeze(dim=1)], 'x_train_f':  ps['xs'][top_indices.squeeze(dim=1)]}
                                self.task_budget[task].append(self.num_train)        
                else:
                    for task in self.params:
                        self.task_budget[task].append(self.task_budget[task][-1])
                # for task in self.nus:
                #     self.task_budget[task].append(self.task_budget[task][-1])
            else:
                for task in self.params:
                    self.task_budget[task].append(self.task_budget[task][-1])
   
   
    def train(self):
        train_av_init_loss = []
        train_av_pde_loss = []
        train_av_tot_loss = []
        eval_metrics = {}
        losses = {}
        epochs = 0
        start = time.time()
        for it in tqdm(range(self.max_iter)):
            self.it = it
            if self.model_type == 'hyper-pinn':
                self.hypernet.train()
            self.network.train()
            if it != 0:
                self.generate_epoch_dataset(it)   
            
            if self.optim == 'lbgfs':
                self.optimizer.step(self.loss_func)
                self.scheduler.step()
                self.StepLR.step()
                for task in self.params:
                    self.train_task_init_loss[task].append(self.train_task_init_loss_val[task].item())
                    self.train_task_pde_loss[task].append(self.train_task_pde_loss_val[task].item())
                    self.train_task_total_loss[task].append(self.train_task_total_loss_val[task].item())
            elif self.optim == 'adam':    
                if self.sampler['type'] == 'self_pace':
                    self.optimizer.zero_grad()
                    self.self_pace_loss()
                    self.optimizer.step()
                else:
                    self.loss_func()
                    self.optimizer.step()
                    # self.StepLR.step()
                # if self.iter < 4500:
                #     self.StepLR.step()
                self.StepLR.step()
            av_l2_loss = self.test(it)
            self.iter += 1
            train_av_init_loss.append(self.u_loss_total.item() / len(self.params))
            train_av_pde_loss.append(self.f_loss_total.item() / len(self.params))
            train_av_tot_loss.append((self.u_loss_total.item() + self.f_loss_total.item()) / len(self.params))
            if av_l2_loss < self.av_l2_loss:
                self.av_l2_loss = av_l2_loss
                torch.save({
                    'epoch': epochs,
                    'model_state_dict': self.hypernet.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, self.output_path + '/best_model.pt')
            elif np.isnan(av_l2_loss):
                print("Early stopping -- loss is NAN")
                return
            epochs += 1
            if epochs == self.max_iter:
                print("Training Finished! To evaluate run stage 2")
            if it % 100 == 0:
                losses['train_task_init_loss'] = self.train_task_init_loss
                losses['train_task_pde_loss'] = self.train_task_pde_loss
                losses['train_task_total_loss'] = self.train_task_total_loss
                losses['train_av_init_loss'] = train_av_init_loss
                losses['train_av_pde_loss'] = train_av_pde_loss
                losses['train_av_total_loss'] = train_av_tot_loss
                losses['prob_per_epoch'] = self.prob_per_epoch
                losses['gi_gsum_per_epoch'] = self.gi_gsum_per_epoch
                losses['l2'] = self.l2_loss
                losses['mse'] = self.mse_loss
                losses['epochs'] = epochs
                # np.save(self.output_path+'/eval.npy', eval_metrics)
                # np.save(self.output_path + '/loss_values.npy', losses)
                torch.save(losses, self.output_path + '/loss_values.pt')
        total = 0
        for task in self.task_budget.keys():
            total += self.task_budget[task][-1]
        self.task_budget['total'] = total
        # np.save(self.output_path + '/budget.npy', self.task_budget)
        torch.save(self.task_budget, self.output_path + '/budget.pt')
        end = time.time()
        print("end time: ", end)
        print("train time: ", end - start)


    def test(self, it):
        self.hypernet.eval()
        av_l2_loss = 0
        for i, task in enumerate(self.params):
            a1, a2 = task
            self.a1, self.a2 = torch.tensor([a1]), torch.tensor([a2])
            self.t_test, self.x_test, self.u_test = Load_ground_truth(self.num_test, self.a1, self.a2, self.device)
            if self.model_type == 'hyper-pinn':
                net_weight, net_bias = self.hypernet(self.a1.to(self.device), self.a2.to(self.device))
                self.network.set_params(net_weight, net_bias)
                latent = None
            elif self.model_type == 'mad-pinn':
                latent = self.network.latent_vector[self.latent_id[task]]
            l2_loss, mse_loss = Test(self.t_test, self.x_test, self.u_test, self.net_u, it, self.output_path, str(a1)+'_'+str(a2), self.num_test, latent)
            self.l2_loss[self.params[i]].append(l2_loss)
            self.mse_loss[self.params[i]].append(mse_loss)
            av_l2_loss += l2_loss
        return av_l2_loss

    def evaluate(self):
        tag = self.tag
        hparams = self.hparams
        output_path = "results/figures/{}".format(tag)
        losses = np.load(output_path + '/loss_values.npy', allow_pickle=True).item()
        prob_per_epoch = losses['prob_per_epoch']
        gigsum_per_epoch = losses['gi_gsum_per_epoch']
        train_av_init_loss = losses['train_av_init_loss']
        train_av_pde_loss = losses['train_av_pde_loss']
        train_av_tot_loss = losses['train_av_total_loss']
        l2_loss = losses['l2']
        mse_loss = losses['mse']
        epochs = len(l2_loss[list(l2_loss.keys())[0]])
        tasks = list(l2_loss.keys())
        selected_tasks = tasks
        utp.plot_single_val_over_epoch(output_path, "average_init_train_loss_fig", epochs, train_av_init_loss, 'epochs',
                                       "loss", "init/boundary loss over epochs")
        utp.plot_single_val_over_epoch(output_path, "average_train_pde_loss_fig", epochs, train_av_pde_loss, 'epochs',
                                       "loss", "pde loss over epochs")
        utp.plot_single_val_over_epoch(output_path, "average_train_total_loss_fig", epochs, train_av_tot_loss, 'epochs',
                                       "loss", "total loss over epochs")
        utp.plot_val_over_epoch(output_path, 'prob_epoch.png', epochs, selected_tasks, prob_per_epoch, 'Epochs',
                                'Probability values', 'Probability values for each task across epochs')
        utp.plot_val_over_epoch(output_path, 'gigsum_epoch.png', epochs, selected_tasks, gigsum_per_epoch, 'Epochs',
                                'gigsum values', 'gigsum values for each task across epochs')
        utp.plot_val_over_epoch(output_path, 'test_l2.png', epochs, selected_tasks, l2_loss, 'Epochs', 'loss values',
                                'l2 loss values for each task across epochs')
        utp.plot_val_over_epoch(output_path, 'test_mse.png', epochs, selected_tasks, mse_loss, 'Epochs', 'loss values',
                                'mse loss values for each task across epochs')
        utp.calculate_disparity(output_path, l2_loss, 1, 'l2')
        utp.calculate_disparity(output_path, l2_loss, 5, 'l2')
        utp.calculate_disparity(output_path, l2_loss, 10, 'l2')
        utp.calculate_disparity(output_path, mse_loss, 1, 'mse')
        utp.calculate_disparity(output_path, mse_loss, 5, 'mse')
        utp.calculate_disparity(output_path, mse_loss, 10, 'mse')
        utp.plot_loss_convergence(output_path, selected_tasks, "mse_conv", mse_loss, hparams.loss['mse_treshold'])
        utp.plot_loss_convergence(output_path, selected_tasks, "l2_conv", l2_loss, hparams.loss['l2_treshold'])
        utp.plot_stacked_bar(output_path, "l2 loss vs weights", l2_loss, prob_per_epoch, selected_tasks)

    def generalization(self):
        checkpoint = torch.load(self.output_path + '/best_model.pt')
        self.hypernet.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.hypernet.eval()
        losses = {}
        self.iter = 100
        self.output_path = self.output_path + '/test_time/'
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        for i, task in enumerate(self.test_params):
            a1, a2 = task
            self.a1, self.a2 = torch.tensor([a1]), torch.tensor([a2])
            self.t_test, self.x_test, self.u_test = Load_ground_truth(self.num_test, self.a1, self.a2, self.device)
            net_weight, net_bias = self.hypernet(self.a1.to(self.device), self.a2.to(self.device))
            self.network.set_params(net_weight, net_bias)
            l2_loss, mse_loss = Test(self.t_test, self.x_test, self.u_test, self.net_u, self.iter, self.output_path,  str(a1) + '_' + str(a2), self.num_test)
            losses[task] = {"l2": l2_loss.item(), "mse": mse_loss.item()}
        with open(self.output_path + '/eval.json', 'w') as handle:
            json.dump(losses, handle)



