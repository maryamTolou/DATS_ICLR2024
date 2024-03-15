import torch
import numpy as np
import torch.utils.data as data
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def Equation(u, t, x, beta):
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    f = u_t + beta * u_x
    return f

def Generate_train_data(num_train, num_ic, num_bc, ic_func, device):
    # collocation points
    t = np.random.rand(num_train, 1)
    x = np.random.rand(num_train, 1)
    x = 2*np.pi*x      # x -> [0, 2*pi], t -> [0, 1]
    t_train_f = torch.tensor(t, requires_grad=True).float()
    x_train_f = torch.tensor(x, requires_grad=True).float()

    # create IC
    t_ic = np.zeros((num_ic, 1))
    x_ic = 2*np.pi*np.random.rand(num_ic, 1)      # x_ic =  0 ~ 2*pi
    u_ic = ic_func(x_ic)
    t_ic_train = torch.tensor(t_ic, requires_grad=True).float()
    x_ic_train = torch.tensor(x_ic, requires_grad=True).float()
    u_ic_train = torch.tensor(u_ic).float()

    # create BC
    tx_bc1 = np.random.rand(num_bc, 2)             # t_bc =  0 ~ 1
    tx_bc1[..., 1] = 2*np.pi                            # x = 2*pi
    tx_bc2 = np.copy(tx_bc1)
    tx_bc2[..., 1] = 0                                   # x = 0
    t_bc1_train = torch.tensor(tx_bc1[..., 0:1]).float()
    x_bc1_train = torch.tensor(tx_bc1[..., 1:2]).float()
    t_bc2_train = torch.tensor(tx_bc2[..., 0:1]).float()
    x_bc2_train = torch.tensor(tx_bc2[..., 1:2]).float()
    return t_train_f.to(device), x_train_f.to(device), t_ic_train.to(device), x_ic_train.to(device), u_ic_train.to(device), t_bc1_train.to(device), x_bc1_train.to(device), t_bc2_train.to(device), x_bc2_train.to(device)


def Load_ground_truth(nu, beta, ic_func):
    number_x = 256
    number_t = 100
    h = 2*np.pi/number_x
    x = np.arange(0, 2*np.pi, h) # not inclusive of the last point
    t = np.linspace(0, 1, number_t).reshape(-1, 1)
    X, T = np.meshgrid(x, t)
    initial_u = ic_func(x)
    source = 0
    F = (np.copy(initial_u)*0)+source # F is the same size as initial_u
    complex_pos = 1j * np.arange(0, number_x/2+1, 1)
    complex_neg = 1j * np.arange(-number_x/2+1, 0, 1)
    complex = np.concatenate((complex_pos, complex_neg))
    complex2 = complex * complex
    initial_uhat = np.fft.fft(initial_u)
    nu_factor = np.exp(nu * complex2 * T - beta * complex * T)
    B = initial_uhat - np.fft.fft(F)*0 # at t=0, second term goes away
    uhat = B*nu_factor + np.fft.fft(F)*T # for constant, fft(p) dt = fft(p)*T
    u = np.real(np.fft.ifft(uhat))
    t_test = t
    x_test = x
    u_test = u
    return t_test, x_test, u_test


def Test(t_test, x_test, u_test, net_u, it, output_path, tag, device, latent):
    t_flat = t_test.flatten()
    x_flat = x_test.flatten()
    t, x = np.meshgrid(t_test, x_test)
    tx = torch.tensor(np.stack([t.flatten(), x.flatten()], axis=-1), requires_grad=True).float().to(device)
    u_pred = net_u(tx[:, 0:1], tx[:, 1:2], latent)
    u_pred = u_pred.detach().cpu().numpy().reshape(t.shape)
    u_gt = u_test.T.reshape(-1, 1)
    l2_loss = np.linalg.norm(u_gt - u_pred.reshape(-1, 1)) / np.linalg.norm(u_gt)
    mse_loss = np.linalg.norm(u_gt - u_pred.reshape(-1, 1))
    if it % 100 == 0:
        print('[Test Iter:%d, Loss: %.5e]' % (it, l2_loss))
        #Plot(it, u_pred, t, x, u_test, t_flat, x_flat, net_u, output_path, tag, device, latent)
    return l2_loss, mse_loss




def Plot(it, u_pred, t, x, u_test, t_flat, x_flat, net_u, output_path, tag, device, latent):
    # image plot
    if it % 10 == 0:
        fig = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 2)
        plt.subplot(gs[0, 0])
        plt.pcolormesh(t, x, u_pred, cmap='rainbow', shading='auto')
        plt.title('Convection', fontsize=20)
        plt.xlabel('t')
        plt.ylabel('x')
        cbar = plt.colorbar(pad=0.05, aspect=10)
        cbar.set_label('u(t,x)')
        cbar.mappable.set_clim(-1, 1)
        plt.subplot(gs[0, 1])
        plt.pcolormesh(t, x, u_test.T, cmap='rainbow', shading='auto')
        plt.xlabel('t')
        plt.ylabel('x')
        cbar = plt.colorbar(pad=0.05, aspect=10)
        cbar.set_label('gt')
        cbar.mappable.set_clim(-1, 1)
        u_pred2 = []
        # plot u(t=const, x) cross-sections
        t_cross_sections = [0, 1.0]
        test_idx = [0, -1]
        for i, t_cs in enumerate(t_cross_sections):
            plt.subplot(gs[1, i])
            t_flat = np.full(x_flat.shape, t_cs).flatten()
            tx = torch.tensor(np.stack([t_flat, x_flat], axis=-1)).float().to(device)
            u_pred = net_u(tx[:, 0:1], tx[:, 1:2], latent)
            u_pred2.append(u_pred.detach().cpu().numpy().reshape(x_flat.shape))
            plt.plot(x_flat, u_test[test_idx[i]], 'b-', linewidth=4, label='Exact')
            plt.plot(x_flat, u_pred2[i], 'r--', linewidth=4, label='Prediction')
            plt.title('t={}'.format(t_cs))
            plt.xlabel('x')
            plt.ylabel('u(t,x)')
        plt.savefig(output_path + "/{}_{}.png".format(tag, it))
        plt.close(fig)