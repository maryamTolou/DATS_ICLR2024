import torch
import numpy as np
import torch.utils.data as data
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def Equation(uvwp, x, y, z, t, a, d):
    def P_function(x, y, z, t, a, d):
        p = - 0.5 * a * a * (torch.exp(2 * a * x) + torch.exp(2 * a * y) + torch.exp(2 * a * z) +
                             2 * torch.sin(a * x + d * y) * torch.cos(a * z + d * x) * torch.exp(a * (y + z)) +
                             2 * torch.sin(a * y + d * z) * torch.cos(a * x + d * y) * torch.exp(a * (z + x)) +
                             2 * torch.sin(a * z + d * x) * torch.cos(a * y + d * z) * torch.exp(a * (x + y))) * \
            torch.exp(-2 * d * d * t)
        return p
    Re = 1
    u = uvwp[:, 0:1]
    v = uvwp[:, 1:2]
    w = uvwp[:, 2:3]
    p = uvwp[:, 3:4]

    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), retain_graph=True, create_graph=True)[0]
    u_zz = torch.autograd.grad(u_z, z, grad_outputs=torch.ones_like(u_z), retain_graph=True, create_graph=True)[0]

    v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
    v_z = torch.autograd.grad(v, z, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), retain_graph=True, create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), retain_graph=True, create_graph=True)[0]
    v_zz = torch.autograd.grad(v_z, z, grad_outputs=torch.ones_like(v_z), retain_graph=True, create_graph=True)[0]

    w_t = torch.autograd.grad(w, t, grad_outputs=torch.ones_like(w), retain_graph=True, create_graph=True)[0]
    w_x = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(w), retain_graph=True, create_graph=True)[0]
    w_y = torch.autograd.grad(w, y, grad_outputs=torch.ones_like(w), retain_graph=True, create_graph=True)[0]
    w_z = torch.autograd.grad(w, z, grad_outputs=torch.ones_like(w), retain_graph=True, create_graph=True)[0]
    w_xx = torch.autograd.grad(w_x, x, grad_outputs=torch.ones_like(w_x), retain_graph=True, create_graph=True)[0]
    w_yy = torch.autograd.grad(w_y, y, grad_outputs=torch.ones_like(w_y), retain_graph=True, create_graph=True)[0]
    w_zz = torch.autograd.grad(w_z, z, grad_outputs=torch.ones_like(w_z), retain_graph=True, create_graph=True)[0]

    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    p_z = torch.autograd.grad(p, z, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]

    f_u = u_t + (u * u_x + v * u_y + w * u_z) + p_x - 1 / Re * (u_xx + u_yy + u_zz)
    f_v = v_t + (u * v_x + v * v_y + w * v_z) + p_y - 1 / Re * (v_xx + v_yy + v_zz)
    f_w = w_t + (u * w_x + v * w_y + w * w_z) + p_z - 1 / Re * (w_xx + w_yy + w_zz)
    f_e = u_x + v_y + w_z
    f_p = p - P_function(x, y, z, t, a, d)
    return f_u, f_v, f_w, f_e, f_p


def Test(tag, net_u, device, epoch, a, d):
    x_star, y_star, z_star, t_star, u_star, v_star, w_star, p_star = Load_ground_truth(device, a, d)
    u_pred, v_pred, w_pred, p_pred = net_u(x_star, y_star, z_star, t_star)
    u_star, v_star, w_star, p_star = u_star.cpu().detach().numpy(), v_star.cpu().detach().numpy(), w_star.cpu().detach().numpy(), p_star.cpu().detach().numpy()
    u_pred, v_pred, w_pred, p_pred = u_pred.cpu().detach().numpy(), v_pred.cpu().detach().numpy(), w_pred.cpu().detach().numpy(), p_pred.cpu().detach().numpy()
    error_u = np.linalg.norm(u_star - u_pred) / np.linalg.norm(u_star)
    error_v = np.linalg.norm(v_star - v_pred) / np.linalg.norm(v_star)
    error_w = np.linalg.norm(w_star - w_pred) / np.linalg.norm(w_star)
    error_p = np.linalg.norm(p_star - p_pred) / np.linalg.norm(p_star)

    # if epoch % 100 == 0:
    #     Plot(tag, net_u, device, epoch, a, d)
    l2_error = (error_u + error_v + error_w + error_p)/4
    mse_error = (np.mean(np.square(u_star - u_pred)) + np.mean(np.square(v_star - v_pred)) +
                np.mean(np.square(w_star - w_pred)) + np.mean(np.square(p_star - p_pred)))/4
    print(str(a) + '-' + '12_Error:%e, mse_Error:%e' % (l2_error, mse_error))
    return l2_error, mse_error

def Plot(tag, net_u, device, epoch, a, d):
    x = np.tile(np.linspace(-1, 1, 51), 51)
    y = np.linspace(-1, 1, 51).repeat(51)
    z = np.array([0.0] * 51 * 51)
    t = np.array([1.0] * 51 * 51)
    u_gt, v_gt, w_gt, p_gt = Analytic_solution(x, y, z, t, a, d)
    x, y, z, t = torch.tensor(x.reshape(-1, 1)).float().to(device), torch.tensor(y.reshape(-1, 1)).float().to(device), torch.tensor(z.reshape(-1, 1)).float().to(device), torch.tensor(t.reshape(-1, 1)).float().to(device)
    u, v, w, p = net_u(x, y, z, t)
    x, y, z, t = x.cpu().detach().numpy(), y.cpu().detach().numpy(), z.cpu().detach().numpy(), t.cpu().detach().numpy()
    u, v, w, p = u.cpu().detach().numpy(), v.cpu().detach().numpy(), w.cpu().detach().numpy(), p.cpu().detach().numpy()
    plt.figure(figsize=(15, 12))
    plt.subplot(3, 3, 1)
    plt.pcolormesh(x.reshape(51, 51), y.reshape(51, 51), u.reshape(51, 51), cmap='rainbow', shading='auto')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('u(t,x)')
    cbar = plt.colorbar(pad=0.05, aspect=10)
    plt.subplot(3, 3, 2)
    plt.pcolormesh(x.reshape(51, 51), y.reshape(51, 51), u_gt.reshape(51, 51), cmap='rainbow', shading='auto')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('gt')
    cbar = plt.colorbar(pad=0.05, aspect=10)
    plt.subplot(3, 3, 3)
    plt.pcolormesh(x.reshape(51, 51), y.reshape(51, 51), np.abs(u_gt.reshape(51, 51) - u.reshape(51, 51)), cmap='rainbow', shading='auto')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('error')
    cbar = plt.colorbar(pad=0.05, aspect=10)

    plt.subplot(3, 3, 4)
    plt.pcolormesh(x.reshape(51, 51), y.reshape(51, 51), v.reshape(51, 51), cmap='rainbow', shading='auto')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('v(t,x)')
    cbar = plt.colorbar(pad=0.05, aspect=10)
    plt.subplot(3, 3, 5)
    plt.pcolormesh(x.reshape(51, 51), y.reshape(51, 51), v_gt.reshape(51, 51), cmap='rainbow', shading='auto')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('gt')
    cbar = plt.colorbar(pad=0.05, aspect=10)
    plt.subplot(3, 3, 6)
    plt.pcolormesh(x.reshape(51, 51), y.reshape(51, 51), np.abs(v_gt.reshape(51, 51) - v.reshape(51, 51)), cmap='rainbow', shading='auto')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('error')
    cbar = plt.colorbar(pad=0.05, aspect=10)

    plt.subplot(3, 3, 7)
    plt.pcolormesh(x.reshape(51, 51), y.reshape(51, 51), w.reshape(51, 51), cmap='rainbow', shading='auto')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('w(t,x)')
    cbar = plt.colorbar(pad=0.05, aspect=10)
    plt.subplot(3, 3, 8)
    plt.pcolormesh(x.reshape(51, 51), y.reshape(51, 51), w_gt.reshape(51, 51), cmap='rainbow', shading='auto')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('gt')
    cbar = plt.colorbar(pad=0.05, aspect=10)
    plt.subplot(3, 3, 9)
    plt.pcolormesh(x.reshape(51, 51), y.reshape(51, 51), np.abs(w_gt.reshape(51, 51) - w.reshape(51, 51)), cmap='rainbow', shading='auto')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('error')
    cbar = plt.colorbar(pad=0.05, aspect=10)
    plt.tight_layout()
    plt.savefig( "NavierStokes/results/figures/{}".format(tag)+ '/'+ str(a)+'-'+str(epoch) + '.png')
    plt.close()


def save_loss_list(problem, loss_list, it, output_path, save_it = 50):
    if problem =='inverse':
        save_it = 5
    if it % save_it ==0:
        np.save(output_path + "/loss_{}.png".format(it), loss_list)


def Analytic_solution(x, y, z, t, a, d):
    u = - a * (np.exp(a * x) * np.sin(a * y + d * z) + np.exp(a * z) * np.cos(a * x + d * y)) * np.exp(- d * d * t)
    v = - a * (np.exp(a * y) * np.sin(a * z + d * x) + np.exp(a * x) * np.cos(a * y + d * z)) * np.exp(- d * d * t)
    w = - a * (np.exp(a * z) * np.sin(a * x + d * y) + np.exp(a * y) * np.cos(a * z + d * x)) * np.exp(- d * d * t)
    p = - 0.5 * a * a * (np.exp(2 * a * x) + np.exp(2 * a * y) + np.exp(2 * a * z) +
                         2 * np.sin(a * x + d * y) * np.cos(a * z + d * x) * np.exp(a * (y + z)) +
                         2 * np.sin(a * y + d * z) * np.cos(a * x + d * y) * np.exp(a * (z + x)) +
                         2 * np.sin(a * z + d * x) * np.cos(a * y + d * z) * np.exp(a * (x + y))) * np.exp(
        -2 * d * d * t)
    return u, v, w, p


def Load_ground_truth(device, a, d):
    x1, y1, z1 = np.linspace(-1, 1, 51), np.linspace(-1, 1, 51), np.linspace(-1, 1, 51)
    x = np.tile(x1, 51 * 51)
    y = np.tile(y1.repeat(51), 51)
    z = z1.repeat(51 * 51)
    # t = np.array([0.5])
    t = np.linspace(0, 1, 11)
    x, y, z = x.repeat(t.shape[0]), y.repeat(t.shape[0]), z.repeat(t.shape[0])
    t = np.tile(t, 51*51*51)
    u, v, w, p = Analytic_solution(x, y, z, t, a, d)
    x, y, z, t = torch.tensor(x.reshape(-1, 1)).float().to(device), torch.tensor(y.reshape(-1, 1)).float().to(device), torch.tensor(z.reshape(-1, 1)).float().to(device), torch.tensor(t.reshape(-1, 1)).float().to(device)
    u, v, w, p = torch.tensor(u.reshape(-1, 1)).float().to(device), torch.tensor(v.reshape(-1, 1)).float().to(device), torch.tensor(w.reshape(-1, 1)).float().to(device), torch.tensor(p.reshape(-1, 1)).float().to(device)
    return x, y, z, t, u, v, w, p


def Generate_train_data(num_train, device, a, d):
    x1, y1, z1 = np.linspace(-1, 1, 51), np.linspace(-1, 1, 51), np.linspace(-1, 1, 51)
    t1 = np.linspace(0, 1, 11)
    b0, b1 = np.array([-1] * 2500), np.array([1] * 2500)
    xt, yt, zt = np.tile(x1[0:50], 50), np.tile(y1[0:50], 50), np.tile(z1[0:50], 50)
    xt1, yt1, zt1 = np.tile(x1[1:51], 50), np.tile(y1[1:51], 50), np.tile(z1[1:51], 50)
    xr, yr, zr = x1[0:50].repeat(50), y1[0:50].repeat(50), z1[0:50].repeat(50)
    xr1, yr1, zr1 = x1[1:51].repeat(50), y1[1:51].repeat(50), z1[1:51].repeat(50)
    train1x = np.concatenate([b1, b0, xt1, xt, xt1, xt], 0).repeat(t1.shape[0])
    train1y = np.concatenate([yt, yt1, b1, b0, yr1, yr], 0).repeat(t1.shape[0])
    train1z = np.concatenate([zr, zr1, zr, zr1, b1, b0], 0).repeat(t1.shape[0])
    train1t = np.tile(t1, 15000)
    train1ub, train1vb, train1wb, train1pb = Analytic_solution(train1x, train1y, train1z, train1t, a, d)
    xb_train = torch.tensor(train1x.reshape(-1, 1)).float()
    yb_train = torch.tensor(train1y.reshape(-1, 1)).float()
    zb_train = torch.tensor(train1z.reshape(-1, 1)).float()
    tb_train = torch.tensor(train1t.reshape(-1, 1)).float()
    ub_train = torch.tensor(train1ub.reshape(-1, 1)).float()
    vb_train = torch.tensor(train1vb.reshape(-1, 1)).float()
    wb_train = torch.tensor(train1wb.reshape(-1, 1)).float()
    x_0 = np.tile(x1, 51 * 51)
    y_0 = np.tile(y1.repeat(51), 51)
    z_0 = z1.repeat(51 * 51)
    t_0 = np.array([0] * x_0.shape[0])
    u_0, v_0, w_0, p_0 = Analytic_solution(x_0, y_0, z_0, t_0, a, d)
    x0_train = torch.tensor(x_0.reshape(-1, 1)).float()
    y0_train = torch.tensor(y_0.reshape(-1, 1)).float()
    z0_train = torch.tensor(z_0.reshape(-1, 1)).float()
    t0_train = torch.tensor(t_0.reshape(-1, 1)).float()
    u0_train = torch.tensor(u_0.reshape(-1, 1)).float()
    v0_train = torch.tensor(v_0.reshape(-1, 1)).float()
    w0_train = torch.tensor(w_0.reshape(-1, 1)).float()
    xx = np.random.randint(31, size=num_train) / 15 - 1
    yy = np.random.randint(31, size=num_train) / 15 - 1
    zz = np.random.randint(31, size=num_train) / 15 - 1
    tt = np.random.randint(11, size=num_train) / 10
    x_train = torch.tensor(xx.reshape(-1, 1), requires_grad=True).float()
    y_train = torch.tensor(yy.reshape(-1, 1), requires_grad=True).float()
    z_train = torch.tensor(zz.reshape(-1, 1), requires_grad=True).float()
    t_train = torch.tensor(tt.reshape(-1, 1), requires_grad=True).float()
    return x0_train.to(device), y0_train.to(device), z0_train.to(device), t0_train.to(device), u0_train.to(device), v0_train.to(device), w0_train.to(device), \
           xb_train.to(device), yb_train.to(device), zb_train.to(device), tb_train.to(device), ub_train.to(device), vb_train.to(device), wb_train.to(device), \
           x_train.to(device), y_train.to(device), z_train.to(device), t_train.to(device)


def Generate_residual_data(num_train, device):
    xx = np.random.randint(31, size=num_train) / 15 - 1
    yy = np.random.randint(31, size=num_train) / 15 - 1
    zz = np.random.randint(31, size=num_train) / 15 - 1
    tt = np.random.randint(11, size=num_train) / 10
    x_train = torch.tensor(xx.reshape(-1, 1), requires_grad=True).float()
    y_train = torch.tensor(yy.reshape(-1, 1), requires_grad=True).float()
    z_train = torch.tensor(zz.reshape(-1, 1), requires_grad=True).float()
    t_train = torch.tensor(tt.reshape(-1, 1), requires_grad=True).float()
    return x_train.to(device), y_train.to(device), z_train.to(device), t_train.to(device)


if __name__ == "__main__":
    b0 = np.array([-1] * 900)
    b1 = np.array([1] * 900)
    x1 = np.linspace(-1, 1, 31)
    y1 = np.linspace(-1, 1, 31)
    z1 = np.linspace(-1, 1, 31)
    t1 = np.linspace(0, 1, 11)
    xt = np.tile(x1[0:30], 30)
    yt = np.tile(y1[0:30], 30)
    zt = np.tile(z1[0:30], 30)
    xt1 = np.tile(x1[1:31], 30)
    yt1 = np.tile(y1[1:31], 30)
    zt1 = np.tile(z1[1:31], 30)
    train1x = np.concatenate([b1, b0, xt1, xt, xt1, xt], 0).repeat(t1.shape[0])
    print(train1x.shape)