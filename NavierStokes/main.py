from network import Base, Hypernet
from model import PINN
from nsutils.params import Params

json_path = 'config/20033.json'
if __name__ == "__main__":
    hparams = Params(json_path)
    stage = hparams.stage
    device = 'cuda:0'
    network = Base().to(device)
    hypernet = Hypernet().to(device)
    model = PINN(network, hypernet, device, hparams, json_path)
    stage = 2
    if stage == 1:
        model.train()
    elif stage == 2:
        model.generalization()
    else:
        NotImplementedError
