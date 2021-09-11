import os
import logging
logger = logging.getLogger('base')
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel
from models.modules.RelModule import RelModule
from models.modules.InvDownscaling import InvDownscaling
from models.modules.InvBlock import InvBlock
from models.modules.CSLayer import CSLayer
from models.modules.Round import Round
from models.modules.Noise import Noise
from models.modules.loss import ReconstructionLoss, FrequencyLoss
import models.lr_scheduler as lr_scheduler
from .InvNetBase import InvNetBase

class InvNet(InvNetBase):
    def __init__(self, opt):
        super(InvNet, self).__init__(opt)
        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        self.is_train = opt['is_train']

        self.train_opt = opt['train']
        self.net_opt = opt['network']

        self.rel_opt = self.net_opt['rel']
        self.down_opt = self.net_opt['down']
        self.block_opt = self.net_opt['block']
        self.lcs_opt = self.net_opt['cs']
        self.quant = self.net_opt['quant']
        self.criterion_opt = self.net_opt['criterions']
        self.lambda_opt = self.net_opt['lambdas']

        self.rel = RelModule(self.rel_opt).to(self.device)
        self.down = InvDownscaling(self.down_opt).to(self.device)
        self.block = InvBlock(self.block_opt).to(self.device)
        self.cs = CSLayer(self.lcs_opt).to(self.device)
        if self.quant == 'round':
            self.quantization = Round()
        elif self.quant == 'noise':
            self.quantization = Noise()
        self.net = nn.ModuleList([self.rel, self.down, self.block, self.cs, self.quantization])
        
        if opt['dist']:
            self.net = DistributedDataParallel(self.net, device_ids=[torch.cuda.current_device()])

        if self.is_train:
            wd = self.train_opt['weight_decay'] if self.train_opt['weight_decay'] else 0 # find train_opt
            optim_params = []
            for sub in self.net:
                sub_optim_params = []
                for k, v in sub.named_parameters():
                    if v.requires_grad:
                        sub_optim_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))
                optim_params.append({'params': sub_optim_params})

            # optimizers
            self.optimizer = torch.optim.Adam(
                optim_params, 
                lr=self.train_opt['lr'], 
                weight_decay=wd, 
                betas=(self.train_opt['beta1'], self.train_opt['beta2'])
            )

            # schedulers
            if self.train_opt['lr_scheme'] == 'MultiStepLR':
                self.scheduler = lr_scheduler.MultiStepLR_Restart(
                    self.optimizer, self.train_opt['lr_steps'],
                    restarts=self.train_opt['restarts'],
                    weights=self.train_opt['restart_weights'],
                    gamma=self.train_opt['lr_gamma'],
                    clear_state=self.train_opt['clear_state']
                )
            elif self.train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                self.scheduler = lr_scheduler.CosineAnnealingLR_Restart(
                    optimizer, self.train_opt['T_period'], 
                    eta_min=self.train_opt['eta_min'],
                    restarts=self.train_opt['restarts'], 
                    weights=self.train_opt['restart_weights']
                )
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')
            
            self.net.train()

            # loss for embedding frame
            if self.criterion_opt['criterion_embedding_frame_basic']:
                self.criterion_embedding_frame_basic = ReconstructionLoss(losstype=self.criterion_opt['criterion_embedding_frame_basic'])
            else:
                self.criterion_embedding_frame_basic = None

            if self.criterion_opt['criterion_embedding_frame_freq']:
                self.criterion_embedding_frame_freq = FrequencyLoss()
            else:
                self.criterion_embedding_frame_freq = None

            # loss for fake frames
            if self.criterion_opt['criterion_fake_frames_basic']:
                self.criterion_fake_frames_basic = ReconstructionLoss(losstype=self.criterion_opt['criterion_fake_frames_basic'])
            else:
                self.criterion_fake_frames_basic = None

            self.log_dict = OrderedDict()

        # self.print_network()
        self.load()

    def feed_data(self, data):
        self.input_frames = data['input_frames'].to(self.device) # input_frames
        self.ref_frame = data['ref_frame'].to(self.device)
        self.vis_flag = data['vis_flag'] # vis_flag

    def loss_and_log(self):
        loss = 0

        # l_embedding_frame
        if self.criterion_embedding_frame_basic is not None:
            l_embedding_frame_basic = self.lambda_opt['lambda_embedding_frame_basic'] * self.criterion_embedding_frame_basic(self.embedding_frame, self.ref_frame.detach())
            loss += l_embedding_frame_basic
            self.log_dict['l_embedding_frame_basic'] = l_embedding_frame_basic.item()

        if self.criterion_embedding_frame_freq is not None:
            l_embedding_frame_freq = self.lambda_opt['lambda_embedding_frame_freq'] * self.criterion_embedding_frame_freq(self.embedding_frame, self.ref_frame.detach())
            loss += l_embedding_frame_freq
            self.log_dict['l_embedding_frame_freq'] = l_embedding_frame_freq.item()

        # l_fake_frames
        num_of_fake_frames = self.fake_frames.size(1) // 3

        if self.criterion_fake_frames_basic is not None:
            l_fake_frames_basic = 0
            for i in range(0, num_of_fake_frames*3, 3):
                fake_frame = self.fake_frames[:, i:i+3, :, :]
                input_frame = self.input_frames[:, i:i+3, :, :]
                l_fake_frames_basic += self.lambda_opt['lambda_fake_frames_basic'] * self.criterion_fake_frames_basic(fake_frame, input_frame.detach()) / num_of_fake_frames
            loss += l_fake_frames_basic
            self.log_dict['l_fake_frames_basic']  = l_fake_frames_basic.item()

        return loss

    def train(self):
        self.net.train()
        self.optimizer.zero_grad()

        # forward
        self.rel_frames = self.rel(self.input_frames)
        self.down_frames = self.down(self.rel_frames)
        self.output_frames = self.block(self.down_frames)
        self.embedding_frame = self.cs(self.output_frames)
        
        # quantization
        self.embedding_frame_prime = self.quantization(self.embedding_frame, train=True)
        
        # backward
        self.output_frames_prime = self.cs(self.embedding_frame_prime, rev=True)
        self.down_frames_prime = self.block(self.output_frames_prime, rev=True)
        self.rel_frames_prime = self.down(self.down_frames_prime, rev=True)
        self.fake_frames = self.rel(self.rel_frames_prime, rev=True)

        # calculate loss
        loss = self.loss_and_log()
        loss.backward()

        if self.train_opt['gradient_clipping']:
            nn.utils.clip_grad_norm_(self.net.parameters(), self.train_opt['gradient_clipping'])

        self.optimizer.step()

    def test(self):
        self.net.eval()
        with torch.no_grad():
            input_frames = self.input_frames

            # forward
            self.rel_frames = self.rel(self.input_frames)
            self.down_frames = self.down(self.rel_frames)
            self.output_frames = self.block(self.down_frames)
            self.embedding_frame = self.cs(self.output_frames)
            
            # quantization
            self.embedding_frame_prime = self.quantization(self.embedding_frame, train=False)
            
            # backward
            self.output_frames_prime = self.cs(self.embedding_frame_prime, rev=True)
            self.down_frames_prime = self.block(self.output_frames_prime, rev=True)
            self.rel_frames_prime = self.down(self.down_frames_prime, rev=True)
            self.fake_frames = self.rel(self.rel_frames_prime, rev=True)

    def embed(self, input_frames):
        input_frames = input_frames.to(self.device)
        self.net.eval()
        with torch.no_grad():
            rel_frames = self.rel(input_frames)
            down_frames = self.down(rel_frames)
            output_frames = self.block(down_frames)
            embedding_frame = self.cs(output_frames)
            embedding_frame_prime = self.quantization(embedding_frame, train=False)
        return embedding_frame_prime.detach()[0].float().cpu()

    def restore(self, embedding_frame_prime):
        embedding_frame_prime = embedding_frame_prime.to(self.device)
        self.net.eval()
        with torch.no_grad():
            output_frames_prime = self.cs(embedding_frame_prime, rev=True)
            down_frames_prime = self.block(output_frames_prime, rev=True)
            rel_frames_prime = self.down(down_frames_prime, rev=True)
            fake_frames = self.rel(rel_frames_prime, rev=True)
        return fake_frames.detach()[0].float().cpu()

    def get_current_logs(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['ref_frame'] = self.ref_frame.detach()[0].float().cpu()
        out_dict['input_frames'] = self.input_frames.detach()[0].float().cpu()
        out_dict['embedding_frame_prime'] = self.embedding_frame_prime.detach()[0].float().cpu()
        out_dict['fake_frames'] = self.fake_frames.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.net)
        if isinstance(self.net, nn.DataParallel) or isinstance(self.net, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.net.__class__.__name__,
                                             self.net.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.net.__class__.__name__)\

        m = 'structure: {}, with parameters: {:,d}'.format(net_struc_str, n)
        if self.rank <= 0:
            logger.info(m)
            logger.info(s)

    def get_network_description(self, network):
        '''Get the string and total parameters of the network'''
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n

    def save(self, iter_label):
        save_filename = '{}.pth'.format(iter_label)
        save_path = os.path.join(self.opt['path']['models'], save_filename)
        state_dict = self.get_current_state_dict()
        torch.save(state_dict, save_path)

    def get_current_state_dict(self):
        if isinstance(self.net, nn.DataParallel) or isinstance(self.net, DistributedDataParallel):
            net = self.net.module
        else:
            net = self.net
        state_dict = net.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        return state_dict

    def load(self):
        load_path = self.opt['path']['pretrain_model']
        if load_path is not None:
            logger.info('Loading model from [{:s}] ...'.format(load_path))
            state_dict = torch.load(load_path)
            self.load_current_state_dict(state_dict)
        state_dict = None

    def load_current_state_dict(self, state_dict):
        if isinstance(self.net, nn.DataParallel) or isinstance(self.net, DistributedDataParallel):
            net = self.net.module
        else:
            net = self.net
        state_dict_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in state_dict.items():
            if k.startswith('module.'):
                state_dict_clean[k[7:]] = v
            else:
                state_dict_clean[k] = v
        net.load_state_dict(state_dict_clean, strict=True)
        state_dict_clean = None
