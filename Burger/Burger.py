import torch
import numpy as np
import torch
import sys
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.io



def Equation(u, t, x, nu):
    """ The pytorch autograd version of calculating residual """
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
    # if problem == 'forward':
    f = u_t + u * u_x - nu * u_xx
    # elif problem == 'inverse':
    #     f = u_t + u * u_x - inverse_lambda * u_xx
    return f


def Generate_train_data(num_train, num_ic, num_bc, device):
    t = np.random.rand(num_train, 1)
    x = np.random.rand(num_train, 1)
    ''' x -> [-1, 1], t -> [0, 1] '''
    x = 2 * x - 1
    t_train_f = torch.tensor(t, requires_grad= True).float()
    x_train_f = torch.tensor(x, requires_grad= True).float()

    # create IC
    t_ic = np.zeros((num_ic, 1))                # t_ic =  0
    x_ic = 2 * np.random.rand(num_ic, 1) - 1  # x_ic = -1 ~ +1

    # create BC
    t_bc = np.random.rand(num_bc, 1) # t_bc =  0 ~ +1
    x_bc = np.random.rand(num_bc, 1) # x_bc = -1 or +1
    x_bc = 2 * np.round(x_bc) - 1

    t_train = torch.tensor(np.concatenate((t_ic, t_bc)), requires_grad= True).float()
    x_train = torch.tensor(np.concatenate((x_ic, x_bc)), requires_grad= True).float()

    # tx_ic = 2 * np.random.rand(num_ic, 2) - 1      # x_ic = -1 ~ +1

    # create output values for IC and BCs
    u_ic = np.sin(-np.pi * x_ic)        # u_ic = -sin(pi*x_ic)
    u_bc = np.zeros((num_bc, 1))        # u_bc = 0
    u_train = torch.tensor(np.concatenate((u_ic, u_bc))).float()

    return t_train_f.to(device), x_train_f.to(device), t_train.to(device), x_train.to(device), u_train.to(device)


def Load_ground_truth(nu):
    data = scipy.io.loadmat('Burger/Data/Burger_full/burg_data_' + str(nu) +'.mat')
    t_test = (data['time'].T)[0:100]
    x_test = data['x'].T
    u_test = (data['solution'].T)[:, 0:100]
    u_gt = np.real((data['solution'].T)[:, 0:100]).T
    x_gt = np.asarray(data['x'].T).flatten()[:, None]
    t_gt = np.asarray((data['time'].T)[0:100]).flatten()[:, None]
    data_u = torch.tensor(u_gt.reshape(-1, 1)).float()
    X, T = np.meshgrid(x_gt, t_gt)
    data_x = torch.tensor(X.reshape(-1, 1), requires_grad=True).float()
    data_t = torch.tensor(T.reshape(-1, 1), requires_grad=True).float()
    return data_x, data_t, data_u, u_test, t_test, x_test


def Test(pde, task, t_test, x_test, u_test, lambda_1, net_u, problem, it, loss_list, output_path, tag, epochs, latent_vector, device):
    t_flat = t_test.flatten()
    x_flat = x_test.flatten()
    t, x = np.meshgrid(t_test, x_test)
    tx = torch.tensor(np.stack([t.flatten(), x.flatten()], axis=-1)).float().to(device)
    u_pred = net_u(tx[:, 0:1], tx[:, 1:2], pde, latent_vector)

    u_pred = u_pred.detach().cpu().numpy().reshape(t.shape)
    u_gt = u_test.reshape(-1, 1)

    l2_loss = np.linalg.norm(u_gt - u_pred.reshape(-1, 1)) / np.linalg.norm(u_gt)
    mse_loss = np.mean(np.square(u_gt-u_pred.reshape(-1, 1)))
    
    # if problem == 'forward':
    if it % 10 == 0:
    #     # logger.error('Iter %d, l2_Loss: %.5e', it+1, l2_loss)
        print('[Task:%.5e Test Iter:%d, 12_Loss: %.5e]' % (task, it, l2_loss))
    loss_list.append(l2_loss)

    # elif problem == 'inverse':
    #     if it % 1 == 0:
    #         print('[Task %e, Test Iter:%d, 12_Loss: %.5e]' % (task, it, l2_loss))
    #         # logger.error('Iter %d, lambda: %.5e, l2_Loss: %.5e', it+1, lambda_1, l2_loss)

    #     loss_list.append(lambda_1.item())

    # save_loss_list(problem, loss_list, it, output_path)

    if (it % 1000 == 0 or it == epochs) and it != 0:
        Plot(pde, it, u_pred, t, x, u_test, t_flat, x_flat, net_u, output_path, tag, latent_vector, device)
        
    return l2_loss, mse_loss


def Plot(pde, it, u_pred, t, x, u_test, t_flat, x_flat, net_u, output_path, tag, latent_vector, device):
    # image plot
    fig = plt.figure(figsize=(20, 8))
    gs = GridSpec(2, 4)
    plt.subplot(gs[0, 0:2])
    plt.pcolormesh(t, x, u_pred, cmap='rainbow', shading='auto')
    plt.xlabel('t')
    plt.ylabel('x')
    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.set_label('u(t,x)')
    cbar.mappable.set_clim(-1, 1)

    plt.subplot(gs[0, 2:4])
    plt.pcolormesh(t, x, u_test, cmap='rainbow', shading='auto')
    plt.xlabel('t')
    plt.ylabel('x')
    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.set_label('gt')
    cbar.mappable.set_clim(-1, 1)

    u_pred2 = []
    # plot u(t=const, x) cross-sections
    t_cross_sections = [0, 0.5, 0.75, 1.0]
    t_idx = [0, int(u_test.T.shape[0] / 2), int(3 * (u_test.T.shape[0] / 4)), -1]

    for i, t_cs in enumerate(t_cross_sections):
        plt.subplot(gs[1, i])
        t_flat = np.full(x_flat.shape, t_cs).flatten()
        tx = torch.tensor(np.stack([t_flat, x_flat], axis=-1)).float().to(device)
        u_pred = net_u(tx[:, 0:1], tx[:, 1:2], pde, latent_vector)
        u_pred2.append(u_pred.detach().cpu().numpy().reshape(x_flat.shape))

        plt.plot(x_flat, u_test.T[t_idx[i]], 'b-', linewidth=2, label='Exact')
        plt.plot(x_flat, u_pred2[i], 'r--', linewidth=2, label='Predict')
        plt.title('t={}'.format(t_cs))
        plt.xlabel('x')
        plt.ylabel('u(t,x)')
    plt.savefig(output_path + "/epoch_figs/{}_{}_fig_.png".format(tag, it))
    plt.close(fig)

