##### Implementation of FedProx #####
##### importing libraries #####
import copy
import os
import random
import argparse
from tkinter.messagebox import NO

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import options.options as option
from torch import utils as vutils
from data import create_dataset, create_dataloader
from solvers import create_solver
from utils import util

torch.backends.cudnn.determinstic = True

parser = argparse.ArgumentParser(description='Train Super Resolution Models in FedProx')
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
client_epochs = opt['fed']['epochs']


##### Create dataset and dataloader #####
for phase, dataset_opt in sorted(opt['datasets'].items()):
    if phase == 'train':
        train_set = create_dataset(dataset_opt)
        train_set_split = vutils.data.random_split(
            train_set, [int(len(train_set) / num_clients) for _ in range(num_clients)])
        train_loaders = [create_dataloader(x, dataset_opt) for x in train_set_split]
        print("=====> Train Dataset: %s" %train_set.name())
        print("=====> Number of image in each client: %d" %len(train_set_split[0]))

        if train_loaders is None:
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
server_solver = create_solver(opt)
model_name = opt['networks']['which_model'].upper()

print('===> Start Train')
print("==================================================")
print("Method: %s || Scale: %d || Total epoch: %d " %(model_name, scale, num_rounds))


##### Create solver log for saving #####
dir_path = server_solver.exp_root
results_save_path = os.path.join(dir_path, 'results.csv')
solver_log = server_solver.get_current_log()
start_epoch = solver_log['round']


##### Helper functions for federated training #####
def LocalUpddate(client_solver, global_solver, dataloader, dataset, local_epoch):
    
    for epoch in range(1, local_epoch+1):

        train_loss = []
        for iter, batch in enumerate(dataloader):
            client_solver.feed_data(batch)        
            iter_loss = client_solver.train_step()

            ##### add a proximal term on loss #####
            if iter > 0:
                w_diff = torch.tensor(0.)
                for w, w_t in zip(client_solver.model.parameters(), global_solver.model.parameters()):
                    w_diff += torch.pow(torch.norm(w - w_t), 2)
                iter_loss += 0.01 / 2. * w_diff.item()

            batch_size = batch['LR'].size(0)
            train_loss.append(iter_loss*batch_size)
   
    ###### Update lr #####
    client_solver.update_learning_rate(epoch)
    return client_solver.model.state_dict(), sum(train_loss)/len(dataset)


def ServerAggregate(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def Test(solver, dataloader):
    psnr_list = []
    ssim_list = []
    val_loss = []

    for iter, batch in enumerate(dataloader):
        solver.feed_data(batch)
        iter_loss = solver.test()
        val_loss.append(iter_loss)

        ##### Calculate psnr/ssim metrics #####
        visuals = solver.get_current_visual()
        psnr, ssim = util.calc_metrics(visuals['SR'], visuals['HR'], crop_border=scale)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
 
    print("PSNR: {:.2f} SSIM: {:.4f}  Loss: {:.3f}".format(
        sum(psnr_list)/len(psnr_list), sum(ssim_list)/len(ssim_list), sum(val_loss)/len(val_loss)))

    return sum(val_loss)/len(val_loss), sum(psnr_list)/len(psnr_list),\
            sum(ssim_list)/len(ssim_list)


##### evaluation on local model before aggregation #####
def Validate(client_solver, val_loader):
    psnr_list = []
    ssim_list = []
    val_loss_list = []

    for iter, batch in enumerate(val_loader):
        client_solver.feed_data(batch)
        iter_loss = client_solver.test()
        val_loss_list.append(iter_loss)

        ##### calculate psnr/ssim metrics #####
        visuals = client_solver.get_current_visual()
        psnr, ssim = util.calc_metrics(visuals['SR'], visuals['HR'], crop_border=scale)
        psnr_list.append(psnr)
        ssim_list.append(ssim)

    print("PSNR: %.2f  SSIM: %.4f" 
    %(sum(psnr_list)/len(psnr_list), sum(ssim_list)/len(ssim_list)))

    return sum(psnr_list)/len(psnr_list), sum(ssim_list)/len(ssim_list)


##### Initializing models #####
net_global = server_solver.model
w_global = net_global.state_dict()

for client_solver in client_solvers:
    client_solver.model.load_state_dict(w_global)


##### Running in FL senaior #####
results = []

best_net = None
best_psnr = None
best_ssim = None
best_epoch = None

for round in range(1, num_rounds+1):

    ##### client sampling #####
    m = max(int(num_selected), 1)
    clients_idx = np.random.choice(range(num_clients), m, replace=False)

    local_loss = []
    w_clients = []
    w_global = None

    ##### local update #####
    with tqdm(total=num_selected, desc='Round: [%d/%d]'%(round, num_rounds), miniters=1) as t:
        for i in clients_idx:

            w_local, loss = LocalUpddate(
                client_solvers[i], server_solver, train_loaders[i], train_set_split[i], client_epochs)

            local_loss.append(copy.deepcopy(loss))
            w_clients.append(copy.deepcopy(w_local))

            t.set_postfix_str('Client Loss: %.6f' %loss)
            t.update()

            ##### evaluation local model before aggregate to server #####
            local_psnr, local_ssim = Validate(client_solvers[i], val_loader)

    ##### Update global weights #####
    w_global = ServerAggregate(w_clients)

    ##### broadcast #####
    net_global.load_state_dict(w_global)

    for client_solver in client_solvers:
        client_solver.model.load_state_dict(w_global)

    ##### print loss #####
    loss_avg = sum(local_loss)/len(local_loss)
    print("[%s]  Round: %d, Average Loss: %.6f" %(val_set.name(), round, loss_avg))
    
    ##### Validating #####
    print('=====> Validating...')    
    val_loss, psnr, ssim = Test(server_solver, val_loader)
    print("\n")
    
    if best_psnr is None or psnr > best_psnr:
        best_net = copy.deepcopy(net_global)
        best_psnr = psnr
        best_ssim = ssim 
        best_epoch = round
        best_save_path = os.path.join(dir_path, 'best_model.pt')
        torch.save(client_solvers[0].model.state_dict(), best_save_path)

    results.append(np.array([round, loss_avg, val_loss, psnr, ssim, best_psnr, best_ssim]))
    final_results = np.array(results)
    final_results = pd.DataFrame(
        final_results, columns=['round', 'loss_avg', 'loss_test', 'psnr', 'ssim', 'best_psnr', 'best_ssim'])
    final_results.to_csv(results_save_path, index=False)

print('Best model, round: {}, psnr: {}, ssim: {}'.format(best_epoch, best_psnr, best_ssim))
print('===> Finished !')
