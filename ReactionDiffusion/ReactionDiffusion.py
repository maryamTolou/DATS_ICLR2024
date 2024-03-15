import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def Equation(u, t, x, nu, rho):
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
    f = u_t - nu * u_xx - rho * u + rho * u ** 2
    return f


def Generate_train_data(num_train, num_ic, num_bc, ic_func, device):
    # collocation points
    t = np.random.rand(num_train, 1)
    x = np.random.rand(num_train, 1)
    x = 2*np.pi*x       # x -> [0, 2*pi], t -> [0, 1]
    t_train_f = torch.tensor(t, requires_grad=True).float()
    x_train_f = torch.tensor(x, requires_grad=True).float()
    # create IC
    t_ic = np.zeros((num_ic, 1))                  # t_ic =  0
    x_ic = 2*np.pi*np.random.rand(num_ic, 1)      # x_ic =  0 ~ 2*pi
    u_ic = ic_func(x_ic)
    t_ic_train = torch.tensor(t_ic, requires_grad=True).float()
    x_ic_train = torch.tensor(x_ic, requires_grad=True).float()
    u_ic_train = torch.tensor(u_ic).float()
    # create BC
    t_bc1 = np.random.rand(num_bc, 1)      # t_bc1 =  0 ~ 1
    x_bc1 = np.ones((num_bc, 1))*2*np.pi   # x_bc1 = 2*pi
    t_bc2 = np.copy(t_bc1)                 # t_bc2 =  0 ~ 1
    x_bc2 = np.zeros((num_bc, 1))          # x_bc2 = 0
    t_bc1_train = torch.tensor(t_bc1, requires_grad=True).float()
    x_bc1_train = torch.tensor(x_bc1, requires_grad=True).float()
    t_bc2_train = torch.tensor(t_bc2, requires_grad=True).float()
    x_bc2_train = torch.tensor(x_bc2, requires_grad=True).float()
    return t_train_f.to(device), x_train_f.to(device), t_ic_train.to(device), x_ic_train.to(device), u_ic_train.to(device), t_bc1_train.to(device), x_bc1_train.to(device), t_bc2_train.to(device), x_bc2_train.to(device)


def reaction(u, rho, dt):
    """ du/dt = rho*u*(1-u)
    """
    A = u * np.exp(rho * dt)
    B = (1 - u)
    u = A / (A+B)
    return u


def diffusion(u, nu, dt, complex2):
    """ du/dt = nu*d2u/dx2
    """
    A = np.exp(nu * complex2 * dt)
    u_hat = np.fft.fft(u)
    u_hat *= A
    u = np.real(np.fft.ifft(u_hat))
    return u


def Load_ground_truth(nu, rho, ic_func):
    number_x = 256
    number_t = 100
    length = 2*np.pi
    T = 1
    dx = length/number_x
    dt = T/number_t
    x = np.arange(0, length, dx) # not inclusive of the last point
    t = np.linspace(0, T, number_t).reshape(-1, 1)
    X, T = np.meshgrid(x, t)
    u = np.zeros((number_x, number_t))
    complex_pos = 1j * np.arange(0, number_x/2+1, 1)
    complex_neg = 1j * np.arange(-number_x/2+1, 0, 1)
    complex = np.concatenate((complex_pos, complex_neg))
    complex2 = complex * complex
    # call u0 this way so array is (n, ), so each row of u should also be (n, )
    initial_u = ic_func(x)
    u[:,0] = initial_u
    u_ = initial_u
    for i in range(number_t-1):
        u_ = reaction(u_, rho, dt)
        u_ = diffusion(u_, nu, dt, complex2)
        u[:, i+1] = u_
    t_test = t
    x_test = x
    u_test = u.T
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
        print(tag + ':[Test Iter:%d, Loss: %.5e]' % (it, l2_loss))
        Plot(it, u_pred, t, x, u_test, t_flat, x_flat, net_u, output_path, tag, device, latent)
    return l2_loss, mse_loss


def Plot(it, u_pred, t, x, u_test, t_flat, x_flat, net_u, output_path, tag, device, latent):
    if it % 50 == 0:
        fig = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 2)
        plt.subplot(gs[0, 0])
        plt.pcolormesh(t, x, u_pred, cmap='rainbow', shading='auto')
        plt.xlabel('t')
        plt.ylabel('x')
        cbar = plt.colorbar(pad=0.05, aspect=10)
        cbar.set_label('u(t,x)')
        plt.subplot(gs[0, 1])
        plt.pcolormesh(t, x, u_test.T, cmap='rainbow', shading='auto')
        plt.xlabel('t')
        plt.ylabel('x')
        cbar = plt.colorbar(pad=0.05, aspect=10)
        cbar.set_label('gt')
        u_pred2 = []
        # plot u(t=const, x) cross-sections
        t_cross_sections = [0, 0.5]
        for i, t_cs in enumerate(t_cross_sections):
            plt.subplot(gs[1, i])
            t_flat = np.full(x_flat.shape, t_cs).flatten()
            tx = torch.tensor(np.stack([t_flat, x_flat], axis=-1)).float().to(device)
            u_pred = net_u(tx[:, 0:1], tx[:, 1:2], latent)
            u_pred2.append(u_pred.detach().cpu().numpy().reshape(x_flat.shape))
            plt.plot(x_flat, u_test[int(t_cross_sections[i]*100)], 'b-', linewidth=4, label='Exact')
            plt.plot(x_flat, u_pred2[i], 'r--', linewidth=4, label='Prediction')
            plt.title('t={}'.format(t_cs))
            plt.xlabel('x')
            plt.ylabel('u(t,x)')
        plt.savefig(output_path + "/{}_{}.png".format(tag, it))
        plt.close(fig)




