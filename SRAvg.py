##### Implementation of FedAvg #####
##### importing libraries #####
import copy
import random, argparse
import torch
import numpy as np
import options.options as option

from tqdm import tqdm
from torch import utils as vutils
from data import create_dataset, create_dataloader
from solvers import create_solver
from utils import util


torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Train Super Resolution Models in FedAVG')
parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
opt = option.parse(parser.parse_args().opt)

##### random seed #####
seed = opt['solver']['manual_seed']
if seed is None: seed = random.randint(1, 10000)
print("=====> Random Seed: %d" %seed)
torch.manual_seed(seed)


##### hyperparameters for federated learning #####
num_clients = opt['fed']['num_clients']
num_selected = int(num_clients * opt['fed']['sample_fraction'])
num_rounds = opt['fed']['num_rounds']
client_epochs = opt['fed']['epochs']


##### Create desired data distribution among clients and dataloader #####
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
global_solver = create_solver(opt)
model_name = opt['networks']['which_model'].upper()


print('===> Start Train')
print("==================================================")
print("Method: %s || Scale: %d || Total round: %d " %(model_name, scale, num_rounds))


##### create solver log for saving #####
solver_log = global_solver.get_current_log()
start_round = solver_log['round']


##### helper function for federated training #####
def Client_Update(client_solver, train_loader, train_set, total_epoch):    
    for epoch in range(1, total_epoch+1):
        train_loss_list = []
        for iter, batch in enumerate(train_loader):
            client_solver.feed_data(batch)
            iter_loss = client_solver.train_step()
            batch_size = batch['LR'].size(0)
            train_loss_list.append(iter_loss*batch_size)


    ###### Update lr #####
    client_solver.update_learning_rate(epoch)
    return client_solver.model.state_dict(), sum(train_loss_list)/len(train_set)


def Server_Aggregate(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] += w[0][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

# def FedAvg(w, weight_avg=None):
#     if weight_avg == None:
#         weight_avg = [1/len(w) for i in range(len(w))]
        
#     w_avg = copy.deepcopy(w[0])
#     for k in w_avg.keys():
#         w_avg[k] = w_avg[k].cuda() * weight_avg[0]
        
#     for k in w_avg.keys():
#         for i in range(1, len(w)):
#             w_avg[k] = w_avg[k].cuda() + w[i][k].cuda() * weight_avg[i]
#     return w_avg




def Test(global_solver, val_loader, solver_log, curr_r):
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
 
    ##### record loss/psnr/ssim #####
    solver_log['records']['val_loss'].append(' ')
    solver_log['records']['val_loss'].append(sum(val_loss_list)/len(val_loss_list))

    solver_log['records']['psnr'].append(' ')
    solver_log['records']['psnr'].append(sum(psnr_list)/len(psnr_list))

    solver_log['records']['ssim'].append(' ')
    solver_log['records']['ssim'].append(sum(ssim_list)/len(ssim_list))

    ##### record the best epoch #####
    round_is_best = False
    if solver_log['best_pred'] < (sum(psnr_list)/len(psnr_list)):
        solver_log['best_pred'] = (sum(psnr_list)/len(psnr_list))
        round_is_best = True
        solver_log['best_round'] = curr_r

    print("PSNR: %.2f  SSIM: %.4f  Loss: %.6f  Best PSNR: %.2f in Round: [%d]" 
    %(sum(psnr_list)/len(psnr_list), sum(ssim_list)/len(ssim_list), sum(val_loss_list)/len(val_loss_list),
      solver_log['best_pred'], solver_log['best_round']))

    global_solver.set_current_log(solver_log)
    global_solver.save_checkpoint(curr_r, round_is_best)
    # global_solver.save_current_log()

    return sum(val_loss_list)/len(val_loss_list), sum(psnr_list)/len(psnr_list),\
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

        ##### Calculate psnr/ssim metrics #####
        visuals = client_solver.get_current_visual()
        psnr, ssim = util.calc_metrics(visuals['SR'], visuals['HR'], crop_border=scale)
        psnr_list.append(psnr)
        ssim_list.append(ssim)

    print("PSNR: %.2f  SSIM: %.4f" 
    %(sum(psnr_list)/len(psnr_list), sum(ssim_list)/len(ssim_list)))

    return sum(psnr_list)/len(psnr_list),sum(ssim_list)/len(ssim_list)


##### Initializing models #####
global_model = global_solver.model
# client_models = [client_solvers[i].model for i in range(num_clients)]

global_w = global_model.state_dict()

for client_solver in client_solvers:
    client_solver.model.load_state_dict(global_w)


##### Running in FL senaior #####
for r in range(1, num_rounds+1):
    ##### select random clients #####
    m = max(int(num_selected), 1)
    clients_idx = np.random.choice(range(num_clients), m, replace=False)

    clients_losses = []
    clients_w = []

    with tqdm(total=num_selected, desc='Round: [%d/%d]'%(r, num_rounds), miniters=1) as t:
        for i in clients_idx:

            w, loss = Client_Update(client_solvers[i], train_loaders[i], train_set_split[i], client_epochs)

            clients_w.append(copy.deepcopy(w))
            clients_losses.append(copy.deepcopy(loss))

            solver_log['records']['client_idx'].append(i)
            solver_log['records']['client_loss'].append(loss)

            t.set_postfix_str('Client Loss: %.6f' %loss)
            t.update()

            ##### evaluation local model before aggregate to server #####
            local_psnr, local_ssim = Validate(client_solvers[i], val_loader)

    # total_data = sum(len(train_set_split[i]) for i in clients_idx)
    # fed_avg_freqs = [len(train_set_split[i]) / total_data for i in clients_idx]

    
    ##### Update global weights #####
    global_w = Server_Aggregate(clients_w)
    # global_w = FedAvg(clients_w, fed_avg_freqs)

    ##### Copy weight to global_net #####
    global_model.load_state_dict(global_w)

    loss_avg = sum(clients_losses)/len(clients_losses)

    solver_log['records']['agg_loss'].append(' ') 
    solver_log['records']['agg_loss'].append(loss_avg)

    print("[%s]  Round: %d, Average Loss: %.6f" %(val_set.name(), r, loss_avg))

    
    ##### Validating #####
    print('=====> Validating...')    
    val_loss, psnr, ssim = Test(global_solver, val_loader, solver_log, r)
    print("\n")
    
    
print('===> Finished !')