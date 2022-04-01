import torch
import torch.nn as nn
from ptflops import get_model_complexity_info


class BaseSolver(object):
    def __init__(self, opt):
        self.opt = opt
        self.scale = opt['scale']
        self.is_train = opt['is_train']
        self.use_chop = opt['use_chop']
        self.self_ensemble = opt['self_ensemble']
        self.use_cl = True if opt['use_cl'] else False

        # GPU verify
        self.use_gpu = torch.cuda.is_available()
        self.Tensor = torch.cuda.FloatTensor if self.use_gpu else torch.FloatTensor

        # for better training (stablization and less GPU memory usage)
        self.last_epoch_loss = 1e8
        self.skip_threshold = opt['solver']['skip_threshold']
        
        # save GPU memory during training
        self.split_batch = opt['solver']['split_batch']

        # experimental dirs
        self.exp_root = opt['path']['exp_root']
        self.checkpoint_dir = opt['path']['epochs']
        self.records_dir = opt['path']['records']
        self.visual_dir = opt['path']['visual']

        # log and vis scheme
        self.save_ckp_step = opt['solver']['save_ckp_step']
        self.save_vis_step = opt['solver']['save_vis_step']

        self.best_round = 0
        self.cur_round = 1
        self.best_pred = 0.0

    def feed_data(self, batch):
        pass

    def train_step(self):
        pass

    def test(self):
        pass

    def _forward_x8(self, x, forward_function):
        pass

    def _overlap_crop_forward(self, upscale):
        pass

    def get_current_log(self):
        pass

    def get_current_visual(self):
        pass

    def get_current_learning_rate(self):
        pass

    def set_current_log(self, log):
        pass

    def update_learning_rate(self, epoch):
        pass

    def save_checkpoint(self, epoch, is_best):
        pass

    def load(self):
        pass

    def save_current_visual(self, epoch, iter):
        pass

    def save_current_log(self):
        pass

    def print_network(self):
        pass

    def get_network_description(self, network):
        '''Get the string and total parameters of the network'''
        if isinstance(network, nn.DataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))

        return s, n


    # def get_network_description(self, network):
    #     '''Get the string, FLOPs and total parameters of the network'''
    #     if isinstance(network, nn.DataParallel):
    #         network = network.module
    #     s = str(network)
    #     n = sum(map(lambda x: x.numel(), network.parameters()))
    #     input = torch.randn(16, 3, 40, 40)
    #     flops, params = get_model_complexity_info(network, (3, 40, 40), as_strings=True, print_per_layer_stat=True, verbose=True)

    #     return s, n, flops
