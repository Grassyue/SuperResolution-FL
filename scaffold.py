##### Implementation of SCAFFOLOD #####
##### importing libraries #####
import copy
import random, argparse
from numpy.core import records
import torch
import numpy as np

from tqdm import tqdm
from solvers import create_solver
from utils import util
from data import create_dataloader, create_dataset
from options import options as option
from torch import utils as vutils

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Super Resolution in SCAFFOLD')
parser.add_argument('-opt', type=str, required=True)
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


##### create dataloader for client and server #####
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


##### create model and solver for client and server #####
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
def train(global_w, c_global, client_solver, train_loader, train_set, total_epoch, c_local):

    cnt = 0

    for epoch in range(1, total_epoch+1):
        train_loss_list = []

        for iter, batch in enumerate(train_loader):
            client_solver.feed_data(batch)
            iter_loss = client_solver.train_step()
            batch_size = batch['LR'].size(0)
            train_loss_list.append(iter_loss * batch_size)

            net_para = client_solver.model.state_dict()
            lr = client_solver.get_current_learning_rate()
            for key in net_para:
                net_para[key] = net_para[key]-lr*(c_global[key]-c_local[key])
            client_solver.model.load_state_dict(net_para)

            cnt += 1

    c_new = copy.deepcopy(c_local)
    c_delta = copy.deepcopy(c_local)
    net_para = client_solver.model.state_dict()    

    for key in net_para:
        c_new[key] = c_new[key]-c_global[key]+(global_w[key]-net_para[key])/(cnt*lr)
        c_delta[key] = c_new[key]-c_local[key]

    c_local = copy.deepcopy(c_new)

    ###### Update lr #####
    client_solver.update_learning_rate(epoch)

    return sum(train_loss_list)/len(train_set), c_delta



def FedAvg(w, weight_avg=None):
    if weight_avg == None:
        weight_avg = [1/len(w) for i in range(len(w))]
        
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w_avg[k].cuda() * weight_avg[0]
        
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] = w_avg[k].cuda() + w[i][k].cuda() * weight_avg[i]
    return w_avg



def Test(global_solver, val_loader, solver_log, current_r):
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
 
    # ##### record loss/psnr/ssim #####
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
        solver_log['best_round'] = current_r


    print("PSNR: %.2f  SSIM: %.4f  Loss: %.6f  Best PSNR: %.2f in Round: [%d]" 
    %(sum(psnr_list)/len(psnr_list), sum(ssim_list)/len(ssim_list), sum(val_loss_list)/len(val_loss_list),
      solver_log['best_pred'], solver_log['best_round']))

    global_solver.set_current_log(solver_log)
    global_solver.save_checkpoint(current_r, round_is_best)
    global_solver.save_current_log()

    return sum(val_loss_list)/len(val_loss_list), sum(psnr_list)/len(psnr_list),\
           sum(ssim_list)/len(ssim_list)



##### Initializing models #####
global_model = global_solver.model
global_w = global_model.state_dict()

for i in range(num_clients):
    client_solvers[i].model.load_state_dict(global_w)
    

initial_state_dict = copy.deepcopy(global_model.state_dict())
server_state_dict = copy.deepcopy(global_model.state_dict())

total = 0
for name, param in global_model.named_parameters():
    total += np.prod(param.size())


c_global = copy.deepcopy(initial_state_dict)
for key in initial_state_dict.keys():
    c_global[key] = torch.zeros_like(initial_state_dict[key])

for idx in range(num_clients):
    c_local = copy.deepcopy(c_global)


##### start training #####
total_train_loss = []
total_val_loss = []
psnr_list = []
ssim_list = []


for r in range(1, num_rounds+1):
    ##### select random clients #####
    m = max(int(num_selected), 1)
    clients_idx = np.random.choice(range(num_clients), m, replace=False)
    clients_losses = []

    total_delta = copy.deepcopy(initial_state_dict)
    for key in total_delta:
        total_delta[key] = torch.zeros_like(initial_state_dict[key])

    
    with tqdm(total=num_selected, desc='Round: [%d/%d]'%(r, num_rounds), miniters=1) as t:
        for i in clients_idx:
            client_solvers[i].model.load_state_dict(copy.deepcopy(global_w))

            loss, c_delta = train(
                global_w, c_global, client_solvers[i], train_loaders[i],
                train_set_split[i], client_epochs, c_local)

            clients_losses.append(copy.deepcopy(loss))
            solver_log['records']['client_idx'].append(i)
            solver_log['records']['client_loss'].append(loss)

            t.set_postfix_str('Client loss: %.6f' %loss)
            t.update()
        
            for key in total_delta:
                total_delta[key] = total_delta[key] + c_delta[key]  
        
    for key in total_delta:
        total_delta[key] /= len(clients_idx)

    for key in c_global:
        if c_global[key].type() == 'torch.LongTensor':
            c_global[key] += total_delta[key].type(torch.LongTensor)
        elif c_global[key].type() == 'torch.cuda.LongTensor':
            c_global[key] += total_delta[key].type(torch.cuda.LongTensor)
        else:
            c_global[key] += total_delta[key]

    w_locals = []
    for i in clients_idx:
        w_locals.append(copy.deepcopy(client_solvers[i].model.state_dict()))

    ww = FedAvg(w_locals)
    global_w = copy.deepcopy(ww)    
    global_model.load_state_dict(global_w)

    loss_avg = sum(clients_losses)/len(clients_losses)
    print("Round: %d, Average Loss: %.6f" %(r, loss_avg))

    solver_log['records']['agg_loss'].append(' ')
    solver_log['records']['agg_loss'].append(loss_avg)


    ##### Validating #####
    print('=====> Validating...')    
    val_loss, psnr, ssim = Test(global_solver, val_loader, solver_log, r)
    print("\n")


print('===> Finished !')
