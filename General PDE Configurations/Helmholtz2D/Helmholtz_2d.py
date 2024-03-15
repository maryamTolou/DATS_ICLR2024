import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def Equation(u, y, x, coefficient):
    u_y = torch.autograd.grad(u, y, torch.ones_like(u), True, True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), True, True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), True, True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), True, True)[0]
    f = u_yy + u_xx + coefficient * u
    return f

def helmholtz_2d_exact_u(y, x, a1, a2):
    return torch.sin(a1*np.pi*y) * torch.sin(a2*np.pi*x)


def helmholtz_2d_source_term(y, x, a1, a2, coefficient):
    u_gt = helmholtz_2d_exact_u(y, x, a1, a2)
    u_yy = -(a1*np.pi)**2 * u_gt
    u_xx = -(a2*np.pi)**2 * u_gt
    return u_yy + u_xx + coefficient*u_gt


def Generate_collocation_points(num_train, device):
    # colocation points
    yc = torch.empty((num_train, 1), dtype=torch.float32).uniform_(-1., 1.)
    xc = torch.empty((num_train, 1), dtype=torch.float32).uniform_(-1., 1.)
    # requires grad
    yc.requires_grad = True
    xc.requires_grad = True
    
    return yc.to(device), xc.to(device)

def Generate_train_data(num_bc, device):
    # boundary points
    north = torch.empty((num_bc, 1), dtype=torch.float32).uniform_(-1., 1.)
    west = torch.empty((num_bc, 1), dtype=torch.float32).uniform_(-1., 1.)
    south = torch.empty((num_bc, 1), dtype=torch.float32).uniform_(-1., 1.)
    east = torch.empty((num_bc, 1), dtype=torch.float32).uniform_(-1., 1.)
    yb = torch.cat([
        torch.ones((num_bc, 1)), west,
        torch.ones((num_bc, 1)) * -1, east
        ]).to(device)
    xb = torch.cat([
        north, torch.ones((num_bc, 1)) * -1,
        south, torch.ones((num_bc, 1))
        ]).to(device)
    # ub = helmholtz_2d_exact_u(yb, xb, a1, a2)
    # return yc.to(device), xc.to(device), uc.to(device), yb.to(device), xb.to(device), ub.to(device)
    return yb.to(device), xb.to(device)


def Test(y_test, x_test, u_test, net_u, it, output_path, tag, num_test, latent):
    u_pred = net_u(y_test, x_test, latent)
    u_pred_arr = u_pred.detach().cpu().numpy()
    u_test_arr = u_test.detach().cpu().numpy()
    l2_loss = np.linalg.norm(u_pred_arr - u_test_arr) / np.linalg.norm(u_test_arr)
    mse_loss = np.linalg.norm(u_pred_arr - u_test_arr)
    if it % 100 == 0:
        print(tag + '[Test Iter:%d, l2_Loss: %.5e]' % (it, l2_loss))
        #if it % 100 == 0:
        #    Plot(it, y_test, x_test, u_pred.detach(), u_test, num_test, output_path, tag)
    return l2_loss, mse_loss


def Plot(it, y, x, u, u_gt, num_test, output_path, tag):
    # ship back to cpu
    y = y.cpu().numpy().reshape(num_test, num_test)
    x = x.cpu().numpy().reshape(num_test, num_test)
    u = u.cpu().numpy().reshape(num_test, num_test)
    u_gt = u_gt.cpu().numpy().reshape(num_test, num_test)
    # plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].set_aspect('equal')
    col0 = axes[0].pcolormesh(x, y, u_gt, cmap='rainbow', shading='auto')
    axes[0].set_xlabel('x', fontsize=12, labelpad=12)
    axes[0].set_ylabel('y', fontsize=12, labelpad=12)
    axes[0].set_title('Exact U', fontsize=18, pad=18)
    div0 = make_axes_locatable(axes[0])
    cax0 = div0.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(col0, cax=cax0)
    axes[1].set_aspect('equal')
    col1 = axes[1].pcolormesh(x, y, u, cmap='rainbow', shading='auto')
    axes[1].set_xlabel('x', fontsize=12, labelpad=12)
    axes[1].set_ylabel('y', fontsize=12, labelpad=12)
    axes[1].set_title('Predicted U', fontsize=18, pad=18)
    div1 = make_axes_locatable(axes[1])
    cax1 = div1.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(col1, cax=cax1)
    axes[2].set_aspect('equal')
    col2 = axes[2].pcolormesh(x, y, np.abs(u - u_gt), cmap='rainbow', shading='auto')
    axes[2].set_xlabel('x', fontsize=12, labelpad=12)
    axes[2].set_ylabel('y', fontsize=12, labelpad=12)
    axes[2].set_title('Absolute error', fontsize=18, pad=18)
    div2 = make_axes_locatable(axes[2])
    cax2 = div2.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(col2, cax=cax2)
    cbar.mappable.set_clim(0, 1)
    plt.tight_layout()
    fig.savefig(output_path + "/{}_{}.png".format(tag, it))
    plt.clf()
    plt.close(fig)


def Load_ground_truth(num_test, a1, a2, device):
    # test points
    y = torch.linspace(-1, 1, num_test)
    x = torch.linspace(-1, 1, num_test)
    # y, x = torch.meshgrid([y, x], indexing='ij')
    y, x = torch.meshgrid([y, x])
    y_test = y.reshape(-1, 1)
    x_test = x.reshape(-1, 1)
    u_test = helmholtz_2d_exact_u(y_test, x_test, a1, a2)
    return y_test.to(device), x_test.to(device), u_test.to(device)