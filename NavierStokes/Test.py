import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def Analytic_solution(x, y, z, t, a , d):
    u = - a * (np.exp(a * x) * np.sin(a * y + d * z) + np.exp(a * z) * np.cos(a * x + d * y)) * np.exp(- d * d * t)
    v = - a * (np.exp(a * y) * np.sin(a * z + d * x) + np.exp(a * x) * np.cos(a * y + d * z)) * np.exp(- d * d * t)
    w = - a * (np.exp(a * z) * np.sin(a * x + d * y) + np.exp(a * y) * np.cos(a * z + d * x)) * np.exp(- d * d * t)
    p = - 0.5 * a * a * (np.exp(2 * a * x) + np.exp(2 * a * y) + np.exp(2 * a * z) +
                         2 * np.sin(a * x + d * y) * np.cos(a * z + d * x) * np.exp(a * (y + z)) +
                         2 * np.sin(a * y + d * z) * np.cos(a * x + d * y) * np.exp(a * (z + x)) +
                         2 * np.sin(a * z + d * x) * np.cos(a * y + d * z) * np.exp(a * (x + y))) * np.exp(
        -2 * d * d * t)
    return u, v, w, p


"""x1, y1, z1 = np.linspace(-1, 1, 31), np.linspace(-1, 1, 31), np.linspace(-1, 1, 31)
t1 = np.linspace(0, 1, 11)
b0, b1 = np.array([-1] * 900), np.array([0] * 900)
xt, yt, zt = np.tile(x1[0:30], 30), np.tile(y1[0:30], 30), np.tile(z1[0:30], 30)
xt1, yt1, zt1 = np.tile(x1[1:31], 30), np.tile(y1[1:31], 30), np.tile(z1[1:31], 30)
xr, yr, zr = x1[0:30].repeat(30), y1[0:30].repeat(30), z1[0:30].repeat(30)
xr1, yr1, zr1 = x1[1:31].repeat(30), y1[1:31].repeat(30), z1[1:31].repeat(30)
train1x = np.concatenate([b1, b0, xt1, xt, xt1, xt], 0).repeat(t1.shape[0])
train1y = np.concatenate([yt, yt1, b1, b0, yr1, yr], 0).repeat(t1.shape[0])
train1z = np.concatenate([zr, zr1, zr, zr1, b1, b0], 0).repeat(t1.shape[0])
train1t = np.tile(t1, 5400)
train1ub, train1vb, train1wb, train1pb = Analytic_solution(train1x, train1y, train1z, train1t)
train1ub = train1ub.reshape(6, -1, 11)"""

x = np.tile(np.linspace(-1, 1, 101), 101)
y = np.linspace(-1, 1, 101).repeat(101)
z = np.array([0.0]*101*101)
t = np.array([1.0]*101*101)
u, v, w, p = Analytic_solution(x, y, z, t, 2, 1)


# gs = GridSpec(2, 2)
plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
plt.pcolormesh(x.reshape(101, 101), y.reshape(101, 101), u.reshape(101, 101), cmap='rainbow', shading='auto')
plt.xlabel('x')
plt.ylabel('y')
plt.title('u(t,x)')
cbar = plt.colorbar(pad=0.05, aspect=10)
# cbar.set_label('u(t,x)')

plt.subplot(1, 3, 2)
plt.pcolormesh(x.reshape(101, 101), y.reshape(101, 101), v.reshape(101, 101), cmap='rainbow', shading='auto')
plt.xlabel('x')
plt.ylabel('y')
plt.title('v(t,x)')
cbar = plt.colorbar(pad=0.05, aspect=10)
# cbar.set_label('v(t,x)')


plt.subplot(1, 3, 3)
plt.pcolormesh(x.reshape(101, 101), y.reshape(101, 101), w.reshape(101, 101), cmap='rainbow', shading='auto')
plt.xlabel('x')
plt.ylabel('y')
plt.title('w(t,x)')
cbar = plt.colorbar(pad=0.05, aspect=10)
# cbar.set_label('w(t,x)')
plt.tight_layout()
"""plt.subplot(2, 2, 4)
plt.pcolormesh(x.reshape(101, 101), y.reshape(101, 101), p.reshape(101, 101), cmap='rainbow', shading='auto')
plt.xlabel('x')
plt.ylabel('y')
plt.title('p(t,x)')
cbar = plt.colorbar(pad=0.05, aspect=10)
# cbar.set_label('p(t,x)')
plt.tight_layout()"""
plt.show()