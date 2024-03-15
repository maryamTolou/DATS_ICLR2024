import os
import torch
from network import *
from utils.grid_sample import *
from model import *
import statistics
from utils.params import Params

config_folder = 'Burger_MAD/'
config_file = '16.json'
json_path = 'Burger/config/' + config_folder + config_file
hparams = Params(json_path)

seed = hparams.seed
np.random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def main(hparams):
    device = hparams.device
    stage = hparams.stage
    model_type = hparams.model_type
    network = Base(hparams).to(device)
    dataset = hparams.pde
    if model_type == "hyper-pinn":
        hypernet = Hypernet(hparams).to(device)
        model = MODEL(network, device, model_type, hypernet, json_path, hparams, dataset)
    elif model_type == "mad-pinn":
        model = MODEL(network, device, model_type, None, json_path, hparams, dataset)
    else:
        print("not implemented")

    if stage == 1:
        model.train()
    elif stage == 2:
        model.evaluate()
    elif stage == 3:
        if model_type == "hyper-pinn":
            model.generalization(device)
        elif model_type == "mad-pinn":
            tag = str(hparams.id) + '_' + str(hparams.seed)
            l2_losses = {task: [] for task in hparams.test_nus}
            mse_losses = {task: [] for task in hparams.test_nus}
            fine_iters = {task: [] for task in hparams.test_nus}
            for task in model.test_nus:
                network = Base(hparams)
                model = MODEL(network, device, model_type, None, json_path, hparams, dataset)
                l2, mse, fine_iter = model.fine_tune(task)
                l2_losses[task] = l2
                mse_losses[task] = mse
                fine_iters[task] = fine_iter
                
            fine_iters["av"] = statistics.mean(list(fine_iters.values()))
            fine_iters["std"] = statistics.stdev(list(fine_iters.values()))
            utp.calculate_disparity(model.output_path, l2_losses, 1, 'l2')
            print(fine_iters)
            with open(model.output_path + "fine_tune_iters", "w") as f:
                json.dump(fine_iters, f)
    else:
        print("1 for training and 2 for evaluation and 3 for generalization")


main(hparams)