import os
import torch
import numpy as np
from model import PINN
from utils.params import Params
from network import Base, Hypernet


json_path = 'config/6000.json'
hparams = Params(json_path)
device = torch.device(hparams.device)
seed = hparams.seed
np.random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    stage = hparams.stage
    network = Base(hparams).to(device)
    hypernet = Hypernet(hparams).to(device)
    pinn = PINN(network, device, hparams.model_type, hypernet, hparams, json_path)
    if stage == 1:
        pinn.train()
    elif stage == 2:
        pinn.evaluate()
    else:
        print("1 for training and 2 for evaluation and 3 for generalization")




