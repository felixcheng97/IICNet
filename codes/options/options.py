import os
import logging
import yaml
from utils.util import OrderedYaml
Loader, Dumper = OrderedYaml()

# type = normal, forward, backward
def parse(opt_path, is_train=True, mode='normal'):
    with open(opt_path, mode='r') as f:
        opt = yaml.load(f, Loader=Loader)

    opt['is_train'] = is_train

    # export CUDA_VISIBLE_DEVICES
    if 'gpu_ids' in opt.keys():
        gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
        print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

    # datasets
    for phase, dataset in opt['datasets'].items():
        dataset['phase'] = phase
        dataset['num_of_frames'] = opt['network']['input']['num_of_frames']
        dataset['scale'] = opt['network']['down']['scale']

    # path
    opt['path']['root'] = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir))
    if is_train:
        experiments_root = os.path.join(opt['path']['root'], 'experiments', opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = os.path.join(experiments_root, 'models')
        opt['path']['training_state'] = os.path.join(experiments_root, 'training_state')
        opt['path']['log'] = experiments_root
        opt['path']['val_samples'] = os.path.join(experiments_root, 'val_samples')

        # change some options for debug mode
        if 'debug' in opt['name']:
            opt['train']['val_freq'] = 8
            opt['logger']['print_freq'] = 1
            opt['logger']['save_checkpoint_freq'] = 8
    else:  # test
        results_root = os.path.join(opt['path']['root'], 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root
        
    # network -> input & embed
    num_of_frames = opt['network']['input']['num_of_frames']
    in_img_nc = opt['network']['input']['in_img_nc']
    emb_img_nc = opt['network']['cs']['out_nc']

    # network -> rel
    opt['network']['rel']['num_of_frames'] = num_of_frames
    
    # network -> down & block
    in_nc = num_of_frames * in_img_nc
    opt['network']['down']['in_nc'] = in_nc
    scale = opt['network']['down']['scale'] if opt['network']['down']['use_down'] else 1
    if scale > 1:
        in_nc = in_nc * scale * scale
        order = opt['network']['down']['order']
        opt['network']['block']['split_len1'] = emb_img_nc * scale * scale if order == 'ref' else in_nc // 4
    else:
        opt['network']['block']['split_len1'] = emb_img_nc
    opt['network']['block']['split_len2'] = in_nc - opt['network']['block']['split_len1']

    # network -> cs
    opt['network']['cs']['in_nc'] = in_nc

    return opt


def dict2str(opt, indent_l=1):
    '''dict to string for logger'''
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg


class NoneDict(dict):
    def __missing__(self, key):
        return None


# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt
