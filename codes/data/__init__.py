'''create dataset and dataloader'''
import logging
import torch
import torch.utils.data


def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
    phase = dataset_opt['phase']
    if phase == 'train':
        if opt['dist']:
            world_size = torch.distributed.get_world_size()
            num_workers = dataset_opt['n_workers']
            assert dataset_opt['batch_size'] % world_size == 0
            batch_size = dataset_opt['batch_size'] // world_size
            shuffle = False
        else:
            num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
            batch_size = dataset_opt['batch_size']
            shuffle = True
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                            num_workers=num_workers, sampler=sampler, drop_last=True,
                                            pin_memory=False)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                                           pin_memory=True)


def create_dataset(dataset_opt):
    mode = dataset_opt['name']
    if mode == 'adobe':
        from data.AdobeDataset import AdobeDataset as D
    elif mode == 'davis':
        from data.DavisDataset import DavisDataset as D
    elif mode == 'flicker1024':
        from data.Flicker1024Dataset import Flicker1024Dataset as D
    elif mode == 'div2kdual':
        from data.Div2kDualDataset import Div2kDualDataset as D
    elif mode == 'div2ksr':
        from data.Div2kSRDataset import Div2kSRDataset as D
    elif mode == 'flicker':
        from data.FlickerDataset import FlickerDataset as D
    elif mode == 'real':
        from data.RealDataset import RealDataset as D
    elif mode == 'voc2012':
        from data.Voc2012Dataset import Voc2012Dataset as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
    dataset = D(dataset_opt)

    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
