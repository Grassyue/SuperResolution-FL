##### importing libraries #####
import copy
import random, argparse
from typing import OrderedDict
import torch
import numpy as np
import options.options as option

from torch import utils as vutils
from data import create_dataset, create_dataloader
from solvers import create_solver
from utils import util


torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Train Super Resolution Models in Federated Senaior')
parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
opt = option.parse(parser.parse_args().opt)

##### Random seed #####
seed = opt['solver']['manual_seed']
if seed is None: seed = random.randint(1, 10000)
print("=====> Random Seed: %d" %seed)
torch.manual_seed(seed)


##### Hyperparameters for federated learning #####
num_clients = opt['fed']['num_clients']
num_selected = int(num_clients * opt['fed']['sample_fraction'])
num_rounds = opt['fed']['num_rounds']
client_epochs = 2


##### Create desired data distribution among clients and dataloader #####
for phase, dataset_opt in sorted(opt['datasets'].items()):
    if phase == 'train':
        train_set = create_dataset(dataset_opt)
        train_set_split = vutils.data.random_split(
            train_set, [int(len(train_set) / num_clients) for _ in range(num_clients)])
        train_loader = [create_dataloader(x, dataset_opt) for x in train_set_split]
        print("=====> Train Dataset: %s" %train_set.name())
        print("=====> Number of image in each client: %d" %len(train_set_split[0]))

        if train_loader is None:
            raise ValueError("[Error] The training data does not exist")

    elif phase == 'val':
        val_set = create_dataset(dataset_opt)
        val_loader = create_dataloader(val_set, dataset_opt)
        print('======> Val Dataset: %s, Number of images: [%d]' %(val_set.name(), len(val_set)))

    else:
        raise NotImplementedError("[Error] Dataset phase [%s] in *.json is not recognized." % phase)


##### Create model and solver #####
scale = opt['scale']
client_solvers = [create_solver(opt) for _ in range(num_clients)]
global_solver = create_solver(opt)
model_name = opt['networks']['which_model'].upper()


print('===> Start Train')
print("==================================================")
print("Method: %s || Scale: %d || Total epoch: %d " %(model_name, scale, client_epochs))



##### Helper functions for federated training #####
def Client_Update(client_solver, train_loader, total_epoch):
    """
    This function updates/trains client model on client data
    """
    
    epoch_loss = []
    for epoch in range(total_epoch):
        batch_loss = []     
        for iter, batch in enumerate(train_loader):
            client_solver.feed_data(batch)
            iter_loss = client_solver.train_step()
            batch_loss.append(iter_loss)
            
        epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        # epoch_loss.append(iter_loss * batch_size)
        # print("Epoch: [%d/%d]  Train Loss: %.6f" %(epoch, client_epochs, sum(epoch_loss)/len(epoch_loss)))

        ###### Update lr #####
        client_solver.update_learning_rate(epoch)

    return client_solver.model.state_dict(), sum(epoch_loss)/len(epoch_loss)


# def Server_Aggregate(w):
#     """
#     This function has aggregation method with 'mean'
#     """
#     w_avg = copy.deepcopy(w[0])
#     for k in w_avg.keys():
#         for i in range(1, len(w)):
#             w_avg[k] += w[i][k] 
#         w_avg[k] = torch.div(w_avg[k], len(w))
#     return w_avg


def Server_Aggregate(w):
    """
    This function has aggregation method with weighted 'mean'
    """

    w_avg = OrderedDict()
    for idx, item in enumerate(w):
        for name, param in item.items():
            if idx == 0:
                w_avg[name] = param * 0.1
            else:
                w_avg[name] += param * 0.1
    return w_avg



def Test(global_solver, val_loader):
    """
    This function val the global model on val data and calculate psnr
    """
    psnr_list = []
    ssim_list = []
    val_loss_list = []

    for iter, batch in enumerate(val_loader):
        global_solver.feed_data(batch)
        iter_loss = global_solver.test()
        val_loss_list.append(iter_loss)

        ##### Calculate psnr/ssim metrics #####
        visuals = global_solver.get_current_visual()
        psnr, ssim = util.calc_metrics(visuals['SR'], visuals['HR'], crop_border=scale)
        psnr_list.append(psnr)
        ssim_list.append(ssim)

    return sum(val_loss_list)/len(val_loss_list), sum(psnr_list)/len(psnr_list), sum(ssim_list)/len(ssim_list)

       
##### Initializing models #####
global_model = global_solver.model
client_models = [client_solvers[i].model for i in range(num_selected)]

global_w = global_model.state_dict()

for model in client_models:
    model.load_state_dict(global_w)


##### Running in FL senaior #####
total_train_loss = []
total_val_loss = []
psnr_list = []
ssim_list = []
net_best = []


for r in range(num_rounds):
    ##### select random clients #####
    m = max(int(num_selected), 0)
    clients_idx = np.random.choice(range(num_clients), m, replace=False)

    clients_losses = []
    clients_w = []

    for i in clients_idx:
        w, loss = Client_Update(client_solvers[i], train_loader[i], client_epochs)
        clients_w.append(copy.deepcopy(w))
        clients_losses.append(copy.deepcopy(loss))   
    
    
    ##### Update global weights #####
    global_w = Server_Aggregate(clients_w)

    ##### Copy weight to global_net #####
    global_model.load_state_dict(global_w)
    

    ##### Testing #####
    print('=====> Validating...')
    val_loss, psnr, ssim = Test(global_solver, val_loader)
    total_val_loss.append(val_loss)
  

    loss_avg = sum(clients_losses)/len(clients_losses)
    total_train_loss.append(loss_avg)

    ##### Save each round psnr and ssim #####
    net_best.append(psnr)


    print("Round: %d, Average Loss: %.6f" %(r, loss_avg))
    print("[%s] PSNR: %.2f  SSIM: %.4f Loss: %.6f" %(val_set.name(), psnr, ssim, sum(total_val_loss)/len(total_val_loss)))
    print("\n")
    
    
print("Best PSNR: %.2f" %(sorted(net_best))[-1])
print('===> Finished !')

