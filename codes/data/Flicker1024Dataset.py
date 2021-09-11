import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import json

class Flicker1024Dataset(Dataset):

    def __init__(self, dataset_opt):
        super(Flicker1024Dataset, self).__init__()
        self.dataset_opt = dataset_opt
        self.phase = dataset_opt['phase']
        self.use_full_frame = dataset_opt['use_full_frame']
        self.frame_size = dataset_opt['frame_size']
        self.annotation_path = dataset_opt['annotation_path']
        self.vis_step = dataset_opt['vis_step']
        self.items = []
        self.item_flags = [] # <- this is for visualization of val and test
        self.process_annotation()

    def process_annotation(self):
        with open(self.annotation_path, 'r') as f:
            data = json.load(f)
            for i, scene in enumerate(sorted(data.keys())):
                for entry, entry_dict in sorted(data[scene].items()):
                    entry_dict = entry_dict.copy()
                    entry_dict['scene'] = scene
                    entry_dict['entry'] = entry
                    if int(scene) % self.vis_step == 0:
                        self.item_flags.append(True)
                    else:
                        self.item_flags.append(False)
                    self.items.append(entry_dict)

    def __getitem__(self, index):
        item = self.items[index]
        item_flag = self.item_flags[index]
        scene = item['scene']
        entry = item['entry']

        input_frames = []
        input_frame_paths = sorted([k for k in item.keys() if 'input_frame' in k])
        for input_frame_path in input_frame_paths:
            input_frames.append(cv2.imread(item[input_frame_path]))
        H = min([input_frame.shape[0] for input_frame in input_frames])
        W = min([input_frame.shape[1] for input_frame in input_frames])
        for i in range(len(input_frame_paths)):
            input_frames[i] = input_frames[i][0:H, 0:W, :]
        input_frames = np.concatenate(input_frames, axis=2)

        ref_frame = cv2.imread(item['ref_frame_path'])

        H, W, _ = input_frames.shape
        if not self.use_full_frame:
            if self.phase == 'train':
                rnd_h = random.randint(0, max(0, H - self.frame_size))
                rnd_w = random.randint(0, max(0, W - self.frame_size))
                input_frames = input_frames[rnd_h:rnd_h + self.frame_size, rnd_w:rnd_w + self.frame_size, :]
                ref_frame = ref_frame[rnd_h:rnd_h + self.frame_size, rnd_w:rnd_w + self.frame_size, :]
            else:
                mid_h = max(0, H - self.frame_size) // 2
                mid_w = max(0, W - self.frame_size) // 2
                input_frames = input_frames[mid_h:mid_h + self.frame_size, mid_w:mid_w + self.frame_size, :]
                ref_frame = ref_frame[mid_h:mid_h + self.frame_size, mid_w:mid_w + self.frame_size, :]
        
        # from numpy to torch.tensor
        input_frames = np.transpose(input_frames, (2, 0, 1))
        ref_frame = np.transpose(ref_frame, (2, 0, 1))

        input_frames = torch.from_numpy(input_frames).float() / 255.0
        ref_frame = torch.from_numpy(ref_frame).float() / 255.0

        return {'input_frames': input_frames, 'ref_frame': ref_frame, 'vis_flag': item_flag, 'scene': scene, 'entry': entry}

    def __len__(self):
        return len(self.items)





