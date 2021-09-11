import os
import os.path as osp
import logging
import time
import argparse
from collections import OrderedDict

import torch
import numpy as np
import options.options as option
import utils.util as util
from data import create_dataset, create_dataloader
from models import create_model

#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, help='Path to options YMAL file.', default='../conf/test/test_iic.yml')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

# for gray scale specific (3 -> 1)
# opt['network']['block']['split_len1'] = 1
# opt['network']['block']['split_len2'] = 2

#### import psnr & ssim functions from util
from utils.util import calculate_ssim
from utils.util import calculate_psnr

import cv2
# cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False)

util.mkdirs(
    (path for key, path in opt['path'].items()
     if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
util.setup_logger('base', opt['path']['log'], opt['name'], level=logging.INFO,
                  screen=True, tofile=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test samples in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

model = create_model(opt)
logger.info('Model parameter numbers: {:d}'.format(sum(p.numel() for p in model.net.parameters() if p.requires_grad)))

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.dataset_opt['name']
    phase = test_loader.dataset.dataset_opt['phase']
    # if test_loader.dataset.opt['use_full_frame']:
    #     test_set_name += '_full'
    logger.info('\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)

    test_metrics = {'embedding_frame_psnr': [], 'fake_frames_psnr': [], 'fake_frames_no_ref_psnr': [],
                    'embedding_frame_ssim': [], 'fake_frames_ssim': [], 'fake_frames_no_ref_ssim': []}
    num_of_frames = opt['network']['input']['num_of_frames']
    for i in range(num_of_frames):
        test_metrics['fake_frame_%02d_psnr' % i] = []
        test_metrics['fake_frame_%02d_ssim' % i] = []

    for data in test_loader:
        scene = data['scene'][0]
        entry = data['entry'][0]
        vis_flag = data['vis_flag'][0].item()
        # vis_flag = True
        this_test_dir = os.path.join(opt['path']['results_root'], test_set_name, phase, scene, entry)
        if vis_flag:
            util.mkdir(this_test_dir)

        # inference
        model.feed_data(data)
        model.test()
        visuals = model.get_current_visuals()

        ref_frame = util.tensor2img(visuals['ref_frame'])
        input_frames = util.tensor2imgs(visuals['input_frames'])
        embedding_frame_prime = util.tensor2img(visuals['embedding_frame_prime'])
        fake_frames = util.tensor2imgs(visuals['fake_frames'])

        num_of_input_frames = input_frames.shape[0]
        num_of_fake_frames = fake_frames.shape[0]

        local_fake_frames_psnr = 0
        local_fake_frames_ssim = 0
        local_fake_frames_no_ref_psnr = 0
        local_fake_frames_no_ref_ssim = 0
        # metrics and save frames
        for i in range(num_of_fake_frames):
            fake_frame_i_psnr = util.calculate_psnr(fake_frames[i], input_frames[i])
            fake_frame_i_ssim = util.calculate_ssim(fake_frames[i], input_frames[i])
            # fake_frame_i_psnr = util.calculate_psnr(cv2.cvtColor(fake_frames[i], cv2.COLOR_BGR2GRAY), cv2.cvtColor(input_frames[i], cv2.COLOR_BGR2GRAY))
            # fake_frame_i_ssim = util.calculate_ssim(cv2.cvtColor(fake_frames[i], cv2.COLOR_BGR2GRAY), cv2.cvtColor(input_frames[i], cv2.COLOR_BGR2GRAY))
            test_metrics['fake_frames_psnr'].append(fake_frame_i_psnr)
            test_metrics['fake_frames_ssim'].append(fake_frame_i_ssim)
            if i != 0:
                test_metrics['fake_frames_no_ref_psnr'].append(fake_frame_i_psnr)
                test_metrics['fake_frames_no_ref_ssim'].append(fake_frame_i_ssim)
            test_metrics['fake_frame_%02d_psnr' % i].append(fake_frame_i_psnr)
            test_metrics['fake_frame_%02d_ssim' % i].append(fake_frame_i_ssim)

            if vis_flag:
                save_input_frame_path = os.path.join(this_test_dir, 'input_frame_{:02d}.png'.format(i+1))
                util.save_img(input_frames[i,...], save_input_frame_path)
                save_fake_frame_path = os.path.join(this_test_dir, 'input_frame_prime_{:02d}_{:.6f}dB_{:.6f}.png'.format(i+1, fake_frame_i_psnr, fake_frame_i_ssim))
                util.save_img(fake_frames[i,...], save_fake_frame_path)

            if i != 0:
                local_fake_frames_no_ref_psnr += fake_frame_i_psnr / (num_of_fake_frames - 1)
                local_fake_frames_no_ref_ssim += fake_frame_i_ssim / (num_of_fake_frames - 1)
            local_fake_frames_psnr += fake_frame_i_psnr / num_of_fake_frames
            local_fake_frames_ssim += fake_frame_i_ssim / num_of_fake_frames
            logger.info('{:20s} - fake frame {:d} psnr: {:.6f} dB; fake frame {:d} ssim: {:.6f}.'.format(scene, i+1, fake_frame_i_psnr, i+1, fake_frame_i_ssim))
        logger.info('{:20s} - fake frames psnr: {:.6f} dB; fake frames ssim: {:.6f}.'.format(scene, local_fake_frames_psnr, local_fake_frames_ssim))
        logger.info('{:20s} - fake frames no ref psnr: {:.6f} dB; fake frames no ref ssim: {:.6f}.'.format(scene, local_fake_frames_no_ref_psnr, local_fake_frames_no_ref_ssim))

        embedding_frame_psnr = util.calculate_psnr(embedding_frame_prime, ref_frame)
        embedding_frame_ssim = util.calculate_ssim(embedding_frame_prime, ref_frame)
        # embedding_frame_psnr = util.calculate_psnr(cv2.cvtColor(embedding_frame_prime, cv2.COLOR_BGR2GRAY), cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY))
        # embedding_frame_ssim = util.calculate_ssim(cv2.cvtColor(embedding_frame_prime, cv2.COLOR_BGR2GRAY), cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY))
        test_metrics['embedding_frame_psnr'].append(embedding_frame_psnr)
        test_metrics['embedding_frame_ssim'].append(embedding_frame_ssim)
        logger.info('{:20s} - embedding frame psnr: {:.6f} dB; embedding frame ssim: {:.6f}.'.format(scene, embedding_frame_psnr, embedding_frame_ssim))

        if vis_flag:
            save_ref_frame_path = os.path.join(this_test_dir, 'ref_frame.png')
            util.save_img(ref_frame, save_ref_frame_path)

    logger.info('----Average PSNR/SSIM results for {}----'.format(test_set_name))
    for i in range(num_of_fake_frames):
        avg_fake_frame_i_psnr = sum(test_metrics['fake_frame_%02d_psnr' % i]) / len(test_metrics['fake_frame_%02d_psnr' % i])
        avg_fake_frame_i_ssim = sum(test_metrics['fake_frame_%02d_ssim' % i]) / len(test_metrics['fake_frame_%02d_ssim' % i])
        logger.info('\tfake frame {:d} psnr: {:.6f} dB; fake frame {:d} ssim: {:.6f}.'.format(i+1, avg_fake_frame_i_psnr, i+1, avg_fake_frame_i_ssim))
    
    avg_fake_frames_psnr = sum(test_metrics['fake_frames_psnr']) / len(test_metrics['fake_frames_psnr'])
    avg_fake_frames_ssim = sum(test_metrics['fake_frames_ssim']) / len(test_metrics['fake_frames_ssim'])
    avg_fake_frames_no_ref_psnr = sum(test_metrics['fake_frames_no_ref_psnr']) / len(test_metrics['fake_frames_no_ref_psnr']) if len(test_metrics['fake_frames_no_ref_psnr']) > 0 else 0
    avg_fake_frames_no_ref_ssim = sum(test_metrics['fake_frames_no_ref_ssim']) / len(test_metrics['fake_frames_no_ref_ssim']) if len(test_metrics['fake_frames_no_ref_ssim']) > 0 else 0
    logger.info('\tfake frames psnr: {:.6f} dB; fake frames ssim: {:.6f}.'.format(avg_fake_frames_psnr, avg_fake_frames_ssim))
    logger.info('\tfake frames no ref psnr: {:.6f} dB; fake frames no ref ssim: {:.6f}.'.format(avg_fake_frames_no_ref_psnr, avg_fake_frames_no_ref_ssim))

    avg_embedding_frame_psnr = sum(test_metrics['embedding_frame_psnr']) / len(test_metrics['embedding_frame_psnr'])
    avg_embedding_frame_ssim = sum(test_metrics['embedding_frame_ssim']) / len(test_metrics['embedding_frame_ssim'])
    logger.info('\tembedding frames psnr: {:.6f} dB; embedding frames ssim: {:.6f}.'.format(avg_embedding_frame_psnr, avg_embedding_frame_ssim))
    

