import os
import math
import argparse
import random
import logging

import torch
import torch.distributed as dist
from data.data_sampler import DistIterSampler

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model

def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YMAL file.', default='./conf/train/train_iic.yml')
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    #### distributed training settings
    rank = args.local_rank
    world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if rank == -1:
        opt['dist'] = False
        print('Disabled distributed training.')
    else:
        opt['dist'] = True

    if world_size > 1:
        torch.cuda.set_device(rank)  # 这里设定每一个进程使用的GPU是一定的
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            util.mkdir_and_rename(
                opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger_val = logging.getLogger('val')  # validation logger
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt['use_tb_logger'] and 'debug' not in opt['name']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir='../tb_logger/' + opt['name'])
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt['dist']:
                train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
                total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info('Number of train samples: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info('Number of val samples in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    #### create model
    model = create_model(opt)

    #### resume training
    if resume_state:
        if rank <= 0: 
            logger.info('Resuming training from epoch: {}, iter: {}.'.format(
                resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
        resume_state = None
    else:
        current_step = 0
        start_epoch = 0

    # torch.cuda.empty_cache()

    #### training
    if rank <= 0: 
        logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
        logger.info('Model parameter numbers: {:d}'.format(sum(p.numel() for p in model.net.parameters() if p.requires_grad)))
    
    for epoch in range(start_epoch, total_epochs + 1):
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            # torch.cuda.empty_cache()
            current_step += 1
            if current_step > total_iters:
                break
            #### training

            model.feed_data(train_data)
            model.train()

            #### update learning rate
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

            #### log
            if current_step % opt['logger']['print_freq'] == 0:
                log = model.get_current_logs()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                    epoch, current_step, model.get_current_learning_rate())
                for k, v in log.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        if rank <= 0:
                            tb_logger.add_scalar('{}'.format(k), v, current_step)
                if rank <= 0:
                    logger.info(message)

            # validation
            if current_step % opt['train']['val_freq'] == 0 and rank <= 0:
                val_metrics = {'embedding_frame_psnr': [], 'fake_frames_psnr': [], 'fake_frames_no_ref_psnr': []}
                num_of_frames = opt['network']['input']['num_of_frames']
                for i in range(num_of_frames):
                    val_metrics['fake_frame_%02d_psnr' % i] = []

                for val_data in val_loader:
                    # torch.cuda.empty_cache()
                    model.feed_data(val_data)
                    model.test()
                    visuals = model.get_current_visuals()

                    ref_frame = util.tensor2img(visuals['ref_frame'])
                    input_frames = util.tensor2imgs(visuals['input_frames'])
                    embedding_frame_prime = util.tensor2img(visuals['embedding_frame_prime'])
                    fake_frames = util.tensor2imgs(visuals['fake_frames'])

                    for i in range(num_of_frames):
                        fake_frame_i_psnr_value = util.calculate_psnr(fake_frames[i], input_frames[i])
                        val_metrics['fake_frames_psnr'].append(fake_frame_i_psnr_value)
                        if i != 0:
                            val_metrics['fake_frames_no_ref_psnr'].append(fake_frame_i_psnr_value)
                        val_metrics['fake_frame_%02d_psnr' % i].append(fake_frame_i_psnr_value)
                    
                    embedding_frame_psnr_value = util.calculate_psnr(embedding_frame_prime, ref_frame)
                    val_metrics['embedding_frame_psnr'].append(embedding_frame_psnr_value)
                    
                    if not val_data['vis_flag']:
                        continue
                    
                    num_of_input_frames = input_frames.shape[0]

                    scene = val_data['scene'][0]
                    entry = val_data['entry'][0]
                    this_val_dir = os.path.join(opt['path']['val_samples'], '{:d}'.format(current_step), scene, entry)
                    util.mkdir(this_val_dir)

                    # save frames
                    for i in range(num_of_input_frames):
                        save_input_frame_path = os.path.join(this_val_dir, 'input_frame_{:02d}.png'.format(i+1))
                        util.save_img(input_frames[i,...], save_input_frame_path)
                        save_fake_frame_path = os.path.join(this_val_dir, 'input_frame_prime_{:02d}.png'.format(i+1))
                        util.save_img(fake_frames[i,...], save_fake_frame_path)

                    save_ref_frame_path = os.path.join(this_val_dir, 'ref_frame.png')
                    util.save_img(ref_frame, save_ref_frame_path)

                    save_embedding_frame_prime_path = os.path.join(this_val_dir, 'embedding_frame_prime.png')
                    util.save_img(embedding_frame_prime, save_embedding_frame_prime_path)

                # log
                avg_embedding_frame_psnr = sum(val_metrics['embedding_frame_psnr']) / len(val_metrics['embedding_frame_psnr'])
                if opt['use_tb_logger'] and 'debug' not in opt['name']:
                    tb_logger.add_scalar('embedding frames psnr', avg_embedding_frame_psnr, current_step)

                avg_fake_frames_psnr = sum(val_metrics['fake_frames_psnr']) / len(val_metrics['fake_frames_psnr'])
                avg_fake_frames_no_ref_psnr = sum(val_metrics['fake_frames_no_ref_psnr']) / len(val_metrics['fake_frames_no_ref_psnr']) if len(val_metrics['fake_frames_no_ref_psnr']) > 0 else 0
                if opt['use_tb_logger'] and 'debug' not in opt['name']:
                    tb_logger.add_scalar('fake frames psnr', avg_fake_frames_psnr, current_step)
                    tb_logger.add_scalar('fake frames no ref psnr', avg_fake_frames_no_ref_psnr, current_step)
                
                avg_fake_frame_i_psnr = []
                for i in range(num_of_frames):
                    avg_fake_frame_i_psnr.append(sum(val_metrics['fake_frame_%02d_psnr' % i]) / len(val_metrics['fake_frame_%02d_psnr' % i]))
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        tb_logger.add_scalar('fake frame {:02d} psnr'.format(i+1), avg_fake_frame_i_psnr[i], current_step)
                
                for i in range(num_of_frames):
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> fake frame {:02d} psnr: {:.6f}.'.format(epoch, current_step, i+1, avg_fake_frame_i_psnr[i]))
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> fake frames psnr: {:.6f}, fake frames no ref psnr: {:.6f}.'.format(epoch, current_step, avg_fake_frames_psnr, avg_fake_frames_no_ref_psnr))
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> embedding frames psnr: {:.6f}.'.format(epoch, current_step, avg_embedding_frame_psnr))
                
                for i in range(num_of_frames):
                    logger.info('# Validation # fake frame {:02d} psnr: {:.6f}.'.format(i+1, avg_fake_frame_i_psnr[i]))
                logger.info('# Validation # fake frames psnr: {:.6f}; fake frames no ref psnr: {:.6f}'.format(avg_fake_frames_psnr, avg_fake_frames_no_ref_psnr))
                logger.info('# Validation # embedding frames psnr: {:.6f}.'.format(avg_embedding_frame_psnr))

                torch.cuda.empty_cache()

            #### save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                if rank <= 0:
                    logger.info('Saving models and training states.')
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)
    if rank <= 0:
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of training.')

if __name__ == '__main__':
    main()

