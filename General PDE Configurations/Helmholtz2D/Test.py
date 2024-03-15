import numpy as np
data = np.load('results/figures/3101_500/loss_values.npy', allow_pickle=True).item()
losses = data['l2']
min_loss = 10
for i in range(10001):
    loss = 0
    for key in losses.keys():
        loss += losses[key][i]
    loss = loss/9
    if loss < min_loss:
        min_loss = loss
print(min_loss)

