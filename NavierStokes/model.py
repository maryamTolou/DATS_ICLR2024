import os
import json
import torch
import shutil
import random
import numpy as np
import nsutils.lr as utr
from tqdm import tqdm
from matplotlib import rc
import matplotlib.pyplot as plt
import time
rc('text', usetex=False)
from NavierStokes import Equation, Test, Load_ground_truth, Generate_train_data, Generate_residual_data
import nsutils.plot as utp

class PINN():
    def __init__(self, network, hypernet, device, hparams, json_path):
        self.hparams = hparams
        self.hypernet = hypernet
        self.network = network
        self.device = device
        self.optimizer = torch.optim.Adam(self.hypernet.parameters())
        self.a2 = 1.0
        self.params = hparams.params
        self.max_iter = 10000
        self.num_train = hparams.train_set['num_train']
        self.optimizer = torch.optim.Adam(self.hypernet.parameters(), lr=self.hparams.lr)
        self.StepLR = utr.CosineAnnealingLR(self.optimizer, T_max=20000, warmup_steps=300)
        self.probs = {}
        for task in self.params:
            self.probs[task] = (torch.ones((1, 1)) * (1 / len(self.params))).item()
        self.sampling_type = hparams.sampler['type']
        self.residual_sampling = hparams.sampler['residual_sampling']
        self.budget = hparams.train_set['num_train'] * len(self.params)
        self.kl_type = hparams.sampler['kl']
        self.eta = hparams.sampler['eta']
        self.prob_period = hparams.sampler['period']
        # for plotting
        self.train_task_total_loss = {task: [] for task in self.params}
        self.l2_loss = {task: [] for task in self.params}
        self.mse_loss = {task: [] for task in self.params}
        self.prob_per_epoch = {task: [] for task in self.params}
        self.gi_gsum_per_epoch = {task: [] for task in self.params}

        self.train_task_total_loss_val = {}
        self.epoch_dataset = {}
        self.scaled_loss_total = torch.tensor([0.0]).to(self.device)
        self.av_l2_loss = 10000
        self.av_mse_loss = 10000
        # output path
        self.tag = str(hparams.id) + '_' + str(hparams.seed)
        self.output_path = "NavierStokes/results/figures/{}".format(self.tag)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        shutil.copy(json_path, self.output_path)
        # train
        self.max_iter = hparams.max_iter
        self.iter = 0
        # test
        self.task_budget = {task: [] for task in self.params}
        for i, task in enumerate(self.params):
            a = task
            self.x0_train, self.y0_train, self.z0_train, self.t0_train, \
            self.u0_train, self.v0_train, self.w0_train, \
            self.xb_train, self.yb_train, self.zb_train, self.tb_train, \
            self.ub_train, self.vb_train, self.wb_train, \
            self.x_train, self.y_train, self.z_train, self.t_train = Generate_train_data(self.num_train, self.device, a, self.a2)
            
            self.x0_val, self.y0_val, self.z0_val, self.t0_val, \
            self.u0_val, self.v0_val, self.w0_val, \
            self.xb_val, self.yb_val, self.zb_val, self.tb_val, \
            self.ub_val, self.vb_val, self.wb_val, \
            self.x_val, self.y_val, self.z_val, self.t_val = Generate_train_data(self.num_train, self.device, a, self.a2)
            
            self.epoch_dataset[task] = {'x_train': self.x_train, 'y_train': self.y_train, 'z_train': self.z_train, 't_train': self.t_train,
                                        'x_val': self.x_val, 'y_val': self.y_val, 'z_val': self.z_val, 't_val': self.t_val}
            self.task_budget[task].append(self.num_train)

    def task_sampler(self):
        gradients_i = {}
        gradients_j = {}
        self.gradient_training = {}
        task_losses_i = {}
        task_losses_j = {}
        probs = {}
      
        for i, task in enumerate(self.params):
            self.optimizer.zero_grad()
            a = task
            self.x0_train, self.y0_train, self.z0_train, self.t0_train, \
            self.u0_train, self.v0_train, self.w0_train, \
            self.xb_train, self.yb_train, self.zb_train, self.tb_train, \
            self.ub_train, self.vb_train, self.wb_train, \
            _, _, _, _ = Generate_train_data(self.num_train, self.device, a, self.a2)
            self.x_train = self.epoch_dataset[task]['x_train']
            self.y_train = self.epoch_dataset[task]['y_train']
            self.z_train = self.epoch_dataset[task]['z_train']
            self.t_train = self.epoch_dataset[task]['t_train']
            scaled_loss = self.compute_task_loss(a, True)
            
            scaled_loss.backward()
            gradient_ = [param.grad.detach().clone() for param in self.hypernet.parameters()]
            self.gradient_training[task] = gradient_
            task_losses_i[task] = scaled_loss.detach()
            
            if self.iter % self.prob_period == 0 and self.iter >= 100:            
                gradient = []
                for g in gradient_:
                    if g is not None:
                        gradient.append(g.detach().view(-1))  # <-- detach added here
                gradient = torch.cat(gradient)
                gradients_i[task] = gradient
                self.optimizer.zero_grad()
                self.x0_val, self.y0_val, self.z0_val, self.t0_val, \
                self.u0_val, self.v0_val, self.w0_val, \
                self.xb_val, self.yb_val, self.zb_val, self.tb_val, \
                self.ub_val, self.vb_val, self.wb_val, \
                self.x_val, self.y_val, self.z_val, self.t_val = Generate_train_data(self.num_train, self.device, a, self.a2)
                self.y_val = self.epoch_dataset[task]['y_val']
                self.x_val = self.epoch_dataset[task]['x_val']
                self.z_val = self.epoch_dataset[task]['z_val']
                self.t_val = self.epoch_dataset[task]['t_val']
                scaled_loss = self.compute_task_loss(a, True)
                scaled_loss.backward(retain_graph=True)
                gradient_ = [param.grad.detach().clone() for param in self.hypernet.parameters()]
                task_losses_j[task] = scaled_loss.detach()
                gradient = []
                for g in gradient_:
                    if g is not None:
                        gradient.append(g.detach().view(-1))  # <-- detach added here
                gradient = torch.cat(gradient)
                gradients_j[task] = gradient

        if self.iter % self.prob_period == 0 and self.iter >= 100:
            for i, task in enumerate(self.params):
                a = task
                if self.kl_type == 'uniform':
                    gigsum = torch.sum(torch.stack([torch.sqrt(task_losses_i[task] * task_losses_j[task2]) * torch.nn.functional.cosine_similarity(
                        gradients_i[task], gradients_j[task2], dim=0) for j, task2 in enumerate(self.params)]))
                    probs[task] = 1/len(self.params) * torch.exp(self.eta * gigsum)
                elif self.kl_type == 'consecutive':
                    gigsum = torch.sum(torch.stack([torch.sqrt(task_losses_i[task] * task_losses_j[task2]) * torch.nn.functional.cosine_similarity(
                            gradients_i[task], gradients_j[task2], dim=0) for j, task2 in enumerate(self.params)]))
                    probs[task] = (self.probs[task] * torch.exp(self.eta * gigsum))
                else:
                    print("KL option is not implemented")
            prob_sum = sum(torch.sum(p) for p in probs.values())
            for task in probs.keys():
                self.probs[task] = (probs[task].item() / prob_sum).item()
        self.gradients_i = gradients_i
        for i, task in enumerate(self.params):
            task_losses_i[task] = task_losses_i[task].detach().item()
        self.task_losses = task_losses_i
    
    def net_u(self, x, y, z, t):
        uvwp = self.network(x, y, z, t, self.lowb, self.upb)
        u = uvwp[:, 0:1]
        v = uvwp[:, 1:2]
        w = uvwp[:, 2:3]
        p = uvwp[:, 3:4]
        return u, v, w, p

    def net_f(self, x, y, z, t):
        uvwp = self.network(x, y, z, t, self.lowb, self.upb)
        f_u, f_v, f_w, f_e, f_p = Equation(uvwp, x, y, z, t, self.a, self.a2)
        return f_u, f_v, f_w, f_e, f_p

    def compute_task_loss(self, a, is_train):
        if is_train:
            self.X0 = torch.cat((self.x0_train, self.y0_train, self.z0_train, self.t0_train), dim=1)
            self.Xb = torch.cat((self.xb_train, self.yb_train, self.zb_train, self.tb_train), dim=1)
            self.X = torch.cat((self.x_train, self.y_train, self.z_train, self.t_train), dim=1)
            self.lowb = self.Xb.min(0)[0]
            self.upb = self.Xb.max(0)[0]
            self.a = torch.tensor([a]).to(self.device)
            net_weight, net_bias = self.hypernet(self.a)
            self.network.set_params(net_weight, net_bias)
            self.u0_pred, self.v0_pred, self.w0_pred, self.p0_pred = self.net_u(self.x0_train, self.y0_train, self.z0_train, self.t0_train)
            self.ub_pred, self.vb_pred, self.wb_pred, self.pb_pred = self.net_u(self.xb_train, self.yb_train, self.zb_train, self.tb_train)
            self.f_u_pred, self.f_v_pred, self.f_w_pred, self.f_e_pred, self.f_p_pred = self.net_f(self.x_train, self.y_train, self.z_train, self.t_train)
            alpha, beta = 100, 100
            scaled_loss = alpha * torch.mean(torch.square(self.u0_pred - self.u0_train)) + \
                        alpha * torch.mean(torch.square(self.v0_pred - self.v0_train)) + \
                        alpha * torch.mean(torch.square(self.w0_pred - self.w0_train)) + \
                        beta * torch.mean(torch.square(self.ub_pred - self.ub_train)) + \
                        beta * torch.mean(torch.square(self.vb_pred - self.vb_train)) + \
                        beta * torch.mean(torch.square(self.wb_pred - self.wb_train)) + \
                        torch.mean(self.f_u_pred**2) + torch.mean(self.f_v_pred**2) + \
                        torch.mean(self.f_w_pred**2) + torch.mean(self.f_e_pred**2) \
                        + alpha * torch.mean(self.f_p_pred**2)
        else:
            self.X0 = torch.cat((self.x0_val, self.y0_val, self.z0_val, self.t0_val), dim=1)
            self.Xb = torch.cat((self.xb_val, self.yb_val, self.zb_val, self.tb_val), dim=1)
            self.X = torch.cat((self.x_val, self.y_val, self.z_val, self.t_val), dim=1)
            self.lowb = self.Xb.min(0)[0]
            self.upb = self.Xb.max(0)[0]
            self.a = torch.tensor([a]).to(self.device)
            net_weight, net_bias = self.hypernet(self.a)
            self.network.set_params(net_weight, net_bias)
            self.u0_pred, self.v0_pred, self.w0_pred, self.p0_pred = self.net_u(self.x0_val, self.y0_val, self.z0_val, self.t0_val)
            self.ub_pred, self.vb_pred, self.wb_pred, self.pb_pred = self.net_u(self.xb_val, self.yb_val, self.zb_val, self.tb_val)
            self.f_u_pred, self.f_v_pred, self.f_w_pred, self.f_e_pred, self.f_p_pred = self.net_f(self.x_val, self.y_val, self.z_val, self.t_val)
            alpha, beta = 100, 100
            scaled_loss = alpha * torch.mean(torch.square(self.u0_pred - self.u0_val)) + \
                        alpha * torch.mean(torch.square(self.v0_pred - self.v0_val)) + \
                        alpha * torch.mean(torch.square(self.w0_pred - self.w0_val)) + \
                        beta * torch.mean(torch.square(self.ub_pred - self.ub_val)) + \
                        beta * torch.mean(torch.square(self.vb_pred - self.vb_val)) + \
                        beta * torch.mean(torch.square(self.wb_pred - self.wb_val)) + \
                        torch.mean(self.f_u_pred**2) + torch.mean(self.f_v_pred**2) + \
                        torch.mean(self.f_w_pred**2) + torch.mean(self.f_e_pred**2) + \
                        alpha * torch.mean(self.f_p_pred**2)
                    
        return scaled_loss

    def loss_func(self):
        self.scaled_loss_total = torch.tensor([0.0]).to(self.device)
        if self.sampling_type == "uniform":
            for i, task in enumerate(self.params):
                a = task
                self.x0_train, self.y0_train, self.z0_train, self.t0_train, \
                self.u0_train, self.v0_train, self.w0_train, \
                self.xb_train, self.yb_train, self.zb_train, self.tb_train, \
                self.ub_train, self.vb_train, self.wb_train, \
                _, _, _, _ = Generate_train_data(self.num_train, self.device, a, self.a2)
                self.x_train = self.epoch_dataset[task]['x_train']
                self.y_train = self.epoch_dataset[task]['y_train']
                self.z_train = self.epoch_dataset[task]['z_train']
                self.t_train = self.epoch_dataset[task]['t_train']
                scaled_loss = self.compute_task_loss(a, True)
                self.train_task_total_loss_val[task] = scaled_loss
                self.scaled_loss_total += scaled_loss
            self.scaled_loss_total.backward()   
                
        else:
            self.task_sampler()
            self.optimizer.zero_grad()
            for i, task in enumerate(self.params):
                a = task
                if self.residual_sampling == False:
                    for p, g in zip(self.hypernet.parameters(), self.gradient_training[task]):
                        p.grad += self.probs[task] * g
                else:
                    for p, g in zip(self.hypernet.parameters(), self.gradient_training[task]):
                        p.grad += g
                self.prob_per_epoch[task].append(self.probs[task])
                self.scaled_loss_total += self.task_losses[task]
        return self.scaled_loss_total


    def generate_epoch_dataset(self, iteration):
        if self.residual_sampling is True and self.sampling_type != 'uniform' and self.iter % self.prob_period ==0:
            for task in self.params:
                num_train = int(self.probs[task]*self.budget)
                num_train = num_train - self.task_budget[task][-1]
                if num_train > 0:
                    x_train, y_train, z_train, t_train = Generate_residual_data(num_train, self.device)
                    self.epoch_dataset[task]['x_train'] = torch.cat((self.epoch_dataset[task]['x_train'], x_train), dim=0)
                    self.epoch_dataset[task]['y_train'] = torch.cat((self.epoch_dataset[task]['y_train'], y_train), dim=0)
                    self.epoch_dataset[task]['z_train'] = torch.cat((self.epoch_dataset[task]['z_train'], z_train), dim=0)
                    self.epoch_dataset[task]['t_train'] = torch.cat((self.epoch_dataset[task]['t_train'], t_train), dim=0)
                    self.task_budget[task].append(num_train + self.task_budget[task][-1])
                elif num_train == 0:
                    self.task_budget[task].append(self.task_budget[task][-1]) 
                else:
                    if self.epoch_dataset[task]['x_train'].shape[0] + num_train <= 10:
                        self.epoch_dataset[task]['x_train'] = self.epoch_dataset[task]['x_train'][0:10]
                        self.epoch_dataset[task]['y_train'] = self.epoch_dataset[task]['y_train'][0:10]
                        self.epoch_dataset[task]['z_train'] = self.epoch_dataset[task]['z_train'][0:10]
                        self.epoch_dataset[task]['t_train'] = self.epoch_dataset[task]['t_train'][0:10]
                        self.task_budget[task].append(10) 
                    else:
                        self.epoch_dataset[task]['x_train'] = self.epoch_dataset[task]['x_train'][np.abs(num_train):self.epoch_dataset[task]['x_train'].shape[0]]
                        self.epoch_dataset[task]['y_train'] = self.epoch_dataset[task]['y_train'][np.abs(num_train):self.epoch_dataset[task]['y_train'].shape[0]]
                        self.epoch_dataset[task]['z_train'] = self.epoch_dataset[task]['z_train'][np.abs(num_train):self.epoch_dataset[task]['z_train'].shape[0]]
                        self.epoch_dataset[task]['t_train'] = self.epoch_dataset[task]['t_train'][np.abs(num_train):self.epoch_dataset[task]['t_train'].shape[0]]
                        self.task_budget[task].append(self.task_budget[task][-1] + num_train)
            for task in self.params:
                if task not in self.params:
                    self.task_budget[task].append(self.task_budget[task][-1])  
        
        else:
            if iteration % self.hparams.residual_sampler['period'] == 0:                             
                if self.hparams.residual_sampler['type'] == 'random':
                    x_train, y_train, z_train, t_train = Generate_residual_data(self.num_train, self.device)
                    for i, task in enumerate(self.params):
                        self.epoch_dataset[task]['x_train'] = x_train
                        self.epoch_dataset[task]['y_train'] = y_train
                        self.epoch_dataset[task]['z_train'] = z_train
                        self.epoch_dataset[task]['t_train'] = t_train
                        self.task_budget[task].append(self.task_budget[task][-1])
                else:
                    for task in self.params:
                        self.task_budget[task].append(self.task_budget[task][-1])
            else:
                for task in self.params:
                    self.task_budget[task].append(self.task_budget[task][-1])

    def train(self):
        train_av_tot_loss = []
        eval_metrics = {}
        losses = {}
        epochs = 0
        
        start = time.time()
        print("start time: ", start )
        epoch_no_loss_improve = 0
        
        for it in tqdm(range(self.max_iter)):
            self.iter = it
            if it != 0:
                self.generate_epoch_dataset(it)
            self.optimizer.zero_grad()
            self.hypernet.train()
            self.network.train()
            self.loss_func()
            self.optimizer.step()
            if self.iter < 4500:
                self.StepLR.step()
            if it % 50 == 0 or it == self.max_iter - 1:
                # print(self.task_budget[0.8][-1], self.task_budget[0.9][-1], self.task_budget[1.0][-1], self.task_budget[1.1][-1], self.task_budget[1.2][-1])
                print(self.epoch_dataset[0.6]['x_train'].shape[0], self.epoch_dataset[0.8]['x_train'].shape[0], self.epoch_dataset[1.0]['x_train'].shape[0],
                      self.epoch_dataset[1.2]['x_train'].shape[0], self.epoch_dataset[1.4]['x_train'].shape[0])
                av_l2_loss, av_mse_loss, disparity_l2, disparity_mse = self.test(it)
            else:
                for i, task in enumerate(self.params):
                    self.l2_loss[task].append(self.l2_loss[task][-1])
                    self.mse_loss[task].append(self.mse_loss[task][-1])
            self.iter += 1
            if av_l2_loss < self.av_l2_loss:
                self.av_l2_loss = av_l2_loss
                self.av_mse_loss = av_mse_loss
                self.disparity_l2 = disparity_l2
                self.disparity_mse = disparity_mse
                # torch.save(self.hypernet.state_dict(), self.output_path + '/best_model.pt')
                torch.save({
                    'epoch': epochs,
                    'model_state_dict': self.hypernet.state_dict(),
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
            torch.cuda.empty_cache()
            del self.scaled_loss_total
        end = time.time()
        print("end time: ", end)
        print("train time: ", end - start)
        losses['prob_per_epoch'] = self.prob_per_epoch
        losses['gi_gsum_per_epoch'] = self.gi_gsum_per_epoch
        losses['l2'] = self.l2_loss
        losses['mse'] = self.mse_loss
        losses['epochs'] = epochs
        with open(self.output_path + '/eval.json', 'w') as handle:
            json.dump(eval_metrics, handle)
        with open(self.output_path + '/loss_values.json', 'w') as handle:
            json.dump(losses, handle)
        total = 0
        for task in self.task_budget.keys():
            total += self.task_budget[task][-1]
        self.task_budget['total'] = total  
        with open(self.output_path + '/budget.json', 'w') as handle:
            json.dump(self.task_budget, handle)
        print('min_average_l2_error:%e, min_average_mse_error:%e, min_disparity_l2:%e, disparity_mse:%e'%(self.av_l2_loss, self.av_mse_loss, self.disparity_l2, self.disparity_mse))

    def test(self, it):
        av_l2_loss = 0
        av_mse_loss = 0
        l2_array = []
        mse_array = []
        self.hypernet.eval()
        for i, task in enumerate(self.params):
            a = task
            self.a = torch.tensor([a]).to(self.device)
            net_weight, net_bias = self.hypernet(self.a)
            self.network.set_params(net_weight, net_bias)
            l2, mse = Test(self.tag, self.net_u, self.device, it, a, self.a2)
            self.l2_loss[task].append(l2)
            self.mse_loss[task].append(mse)
            l2_array.append(l2)
            mse_array.append(mse)
            av_l2_loss += l2
            av_mse_loss += mse
        disparity_l2 = max(l2_array) - min(l2_array)
        disparity_mse = max(mse_array) - min(mse_array)
        return av_l2_loss / len(self.params), av_mse_loss/len(self.params), disparity_l2, disparity_mse

    def generalization(self):
        checkpoint = torch.load(self.output_path + '/best_model.pt', map_location=self.device)
        self.hypernet.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.hypernet.eval()
        self.test_params = [0.7, 0.9, 1.1, 1.3]
        av_l2_loss = 0
        av_mse_loss = 0
        l2_array = []
        mse_array = []
        for i, task in enumerate(self.test_params):
            a = task
            self.a = torch.tensor([a]).to(self.device)
            net_weight, net_bias = self.hypernet(self.a)
            self.network.set_params(net_weight, net_bias)
            self.x0_train, self.y0_train, self.z0_train, self.t0_train, \
            self.u0_train, self.v0_train, self.w0_train, \
            self.xb_train, self.yb_train, self.zb_train, self.tb_train, \
            self.ub_train, self.vb_train, self.wb_train, \
            self.x_train, self.y_train, self.z_train, self.t_train = Generate_train_data(self.num_train, self.device, a,self.a2)
            self.X0 = torch.cat((self.x0_val, self.y0_val, self.z0_val, self.t0_val), dim=1)
            self.Xb = torch.cat((self.xb_val, self.yb_val, self.zb_val, self.tb_val), dim=1)
            self.X = torch.cat((self.x_val, self.y_val, self.z_val, self.t_val), dim=1)
            self.lowb = self.Xb.min(0)[0]
            self.upb = self.Xb.max(0)[0]
            l2, mse = Test(self.tag+'_test', self.net_u, self.device, 0, a, self.a2)
            l2_array.append(l2)
            mse_array.append(mse)
            av_l2_loss += l2
            av_mse_loss += mse
        disparity_l2 = max(l2_array) - min(l2_array)
        disparity_mse = max(mse_array) - min(mse_array)
        print('average_l2_error:%e, average_mse_error:%e, disparity_l2:%e, disparity_mse:%e' % (
        av_l2_loss / len(self.test_params), av_mse_loss/len(self.test_params), disparity_l2, disparity_mse))
                    
    def evaluate(self):
        tag = self.tag
        hparams = self.hparams
        

        output_path = "NavierStokes/results/figures/{}".format(self.tag)  

        with open(output_path + '/loss_values.json', 'r') as handle:
            losses = json.load(handle)
            
        prob_per_epoch = losses['prob_per_epoch'] 
        # gigsum_per_epoch = losses['gi_gsum_per_epoch']

        
        l2_vals = losses['l2']
        epochs = len(l2_vals[list(l2_vals.keys())[0]])
        
        nus = list(l2_vals.keys())
        
        l2_loss = {}
        for task in l2_vals.keys():
            l2_loss[task] = l2_vals[task][-1]
        with open(self.output_path + '/l2_loss.json', 'w') as handle:
            json.dump(l2_loss, handle)

        selected_nus = nus
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.params)))
        utp.plot_val_over_epoch(output_path, 'prob_epoch.png', epochs, selected_nus, prob_per_epoch, colors, "Epochs", "Probability values", "Probability values for each task across epochs")
        # utp.plot_val_over_epoch(output_path, 'gigsum_epoch.png', epochs, selected_nus, gigsum_per_epoch, colors, 'Epochs', 'gigsum values','gigsum values for each task across epochs')
        
        utp.plot_val_over_epoch(output_path, 'test_l2.png', epochs, selected_nus, l2_vals, colors, 'Epochs', 'loss values','l2 loss values for each task across epochs')

        
        utp.calculate_disparity(output_path, l2_vals, 1, 'l2')
        utp.calculate_disparity(output_path, l2_vals, 5, 'l2')
        utp.calculate_disparity(output_path, l2_vals, 10, 'l2')
        
        # utp.plot_stacked_bar(output_path, "l2 loss vs weights", l2_vals, prob_per_epoch, selected_nus)
   
