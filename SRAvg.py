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
print("Method: %s || Scale: %d || Total epoch: %d " %(model_name, scale, client_epochs))



##### Create solver log for saving #####
solver_log = global_solver.get_current_log()
start_epoch = solver_log['epoch']



##### Helper functions for federated training #####
def Client_Update(client_solver, train_loader, train_set, total_epoch, solver_log):
    """
    This function updates/trains client model on client data
    """
    
    for epoch in range(1, total_epoch+1):
        print('\n===> Training Epoch: [%d/%d]  Leaning Rate: %f'
        %(epoch, client_epochs, client_solver.get_current_learning_rate()))

        train_loss_list = []
        with tqdm(total=len(train_loader), desc='Epoch: [%d/%d]'%(epoch, total_epoch), miniters=1) as t:
            for iter, batch in enumerate(train_loader):
                client_solver.feed_data(batch)        
                iter_loss = client_solver.train_step()
                batch_size = batch['LR'].size(0)
                train_loss_list.append(iter_loss*batch_size)
                t.set_postfix_str('Batch Loss: %.4f' %iter_loss)
                t.update()

    print("Epoch: [%d/%d]  Train Loss: %.6f" %(epoch, client_epochs, sum(train_loss_list)/len(train_set)))    
   
    ###### Update lr #####
    client_solver.update_learning_rate(epoch)
    return client_solver.model.state_dict(), sum(train_loss_list)/len(train_set)



def Server_Aggregate(w):
    """
    This function has aggregation method with 'mean'
    """
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] += w[0][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg



def Test(global_solver, val_loader, solver_log, current_r):
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
 
    ##### record loss/psnr/ssim #####
    solver_log['records']['train_loss'].append(sum(val_loss_list)/len(val_loss_list))
    solver_log['records']['val_loss'].append(sum(val_loss_list)/len(val_loss_list))
    solver_log['records']['psnr'].append(sum(psnr_list)/len(psnr_list))
    solver_log['records']['ssim'].append(sum(ssim_list)/len(ssim_list))

    ##### record the best epoch #####
    epoch_is_best = False
    if solver_log['best_pred'] < (sum(psnr_list)/len(psnr_list)):
        solver_log['best_pred'] = (sum(psnr_list)/len(psnr_list))
        epoch_is_best = True
        solver_log['best_epoch'] = current_r

    print("PSNR: %.2f  SSIM: %.4f  Loss: %.6f  Best PSNR: %.2f in Epoch: [%d]" 
    %(sum(psnr_list)/len(psnr_list), sum(ssim_list)/len(ssim_list), sum(val_loss_list)/len(val_loss_list),
      solver_log['best_pred'], solver_log['best_epoch']))

    global_solver.set_current_log(solver_log)
    global_solver.save_checkpoint(current_r, epoch_is_best)
    # global_solver.save_current_log()

    return sum(val_loss_list)/len(val_loss_list), sum(psnr_list)/len(psnr_list),\
           sum(ssim_list)/len(ssim_list)



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
        print("Client idx: %d  Round: %d" %(i, r))
        w, loss = Client_Update(client_solvers[i], train_loaders[i], train_set_split[0], client_epochs, solver_log)
        clients_w.append(copy.deepcopy(w))
        clients_losses.append(copy.deepcopy(loss))   
    
    
    solver_log['records']['lr'].append(global_solver.get_current_learning_rate())
    ##### Update global weights #####
    global_w = Server_Aggregate(clients_w)

    ##### Copy weight to global_net #####
    global_model.load_state_dict(global_w)

    loss_avg = sum(clients_losses)/len(clients_losses)
    print("[%s]  Round: %d, Average Loss: %.6f" %(val_set.name(), r, loss_avg))

    
    ##### Validating #####
    print('=====> Validating...')    
    val_loss, psnr, ssim = Test(global_solver, val_loader, solver_log, r)

    # total_val_loss.append(val_loss)   
    # total_train_loss.append(loss_avg)
    # print("[%s] PSNR: %.2f  SSIM: %.4f Loss: %.6f" 
    #     %(val_set.name(), psnr, ssim, sum(total_val_loss)/len(total_val_loss)))
    print("\n")
    
    
print('===> Finished !')
