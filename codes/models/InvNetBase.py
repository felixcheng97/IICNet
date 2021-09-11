import os
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel


class InvNetBase():
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        # self.is_train = opt['is_train']
        self.blocks = []
        self.scheduler = None
        self.optimizer = None

    def feed_data(self, data):
        pass

    def get_current_visuals(self):
        pass

    def get_current_logs(self):
        pass

    def print_network(self):
        pass

    def save(self, label):
        pass

    def load(self):
        pass

    def _set_lr(self, lr_groups):
        ''' set learning rate for warmup,
        lr_groups_l: list for lr_groups. each for a optimizer'''
        for param_group, lr in zip(self.optimizer.param_groups, lr_groups):
            param_group['lr'] = lr

    def _get_init_lr(self):
        # get the initial lr, which is set by the scheduler
        return [v['initial_lr'] for v in self.optimizer.param_groups]

    def update_learning_rate(self, cur_iter, warmup_iter=-1):
        self.scheduler.step()
        #### set up warm up learning rate
        if cur_iter < warmup_iter:
            # get initial lr for each group
            init_lr = self._get_init_lr()
            # modify warming-up learning rates
            warn_up_lr = [v / warmup_iter * cur_iter for v in init_lr]
            # set learning rate
            self._set_lr(warm_up_lr)

    def get_current_learning_rate(self):
        return self.optimizer.param_groups[0]['lr']

    def save_training_state(self, epoch, iter_step):
        '''Saves training state during training, which will be used for resuming'''
        state = {'epoch': epoch, 'iter': iter_step, 'scheduler': self.scheduler.state_dict(), 'optimizer': self.optimizer.state_dict()}
        save_filename = '{}.state'.format(iter_step)
        save_path = os.path.join(self.opt['path']['training_state'], save_filename)
        torch.save(state, save_path)

    def resume_training(self, resume_state):
        '''Resume the optimizers and schedulers for training'''
        self.scheduler.load_state_dict(resume_state['scheduler'])
        self.optimizer.load_state_dict(resume_state['optimizer'])