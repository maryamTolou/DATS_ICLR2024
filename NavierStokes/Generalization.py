import torch
from network import Base, Hypernet


if __name__ == "__main__":
    device = 'cuda:0'
    model_file = torch.load('NavierStokes/results/20013_500/best_model.pt', map_location=device)['model_state_dict']
    network = Base().to(device)
    hypernet = Hypernet().to(device)
    hypernet.load_state_dict(model_file)
    params = [0.7, 0.9, 1.1, 1.3]
    for i, task in enumerate(params):
        a = torch.tensor([task]).to(device)
        net_weight, net_bias = hypernet(a)
        network.set_params(net_weight, net_bias)



