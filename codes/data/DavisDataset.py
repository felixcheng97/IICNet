import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import json

class DavisDataset(Dataset):

    def __init__(self, dataset_opt):
        super(DavisDataset, self).__init__()
        self.dataset_opt = dataset_opt
        self.phase = dataset_opt['phase']
        self.scale = dataset_opt['scale']
        self.use_full_frame = dataset_opt['use_full_frame']
        self.frame_size = dataset_opt['frame_size']
        self.annotation_path = dataset_opt['annotation_path']
        self.items = []
        self.item_flags = [] # <- this is for visualization of val and test
        self.process_annotation()

    def process_annotation(self):
        with open(self.annotation_path, 'r') as f:
            data = json.load(f)
            for i, scene in enumerate(sorted(data.keys())):
                all_scene_items = sorted(data[scene].items())
                for entry, entry_dict in all_scene_items:
                    entry_dict = entry_dict.copy()
                    entry_dict['scene'] = scene
                    entry_dict['entry'] = entry
                    if int(entry) == 0:
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

        if self.scale > 1:
            H, W, _ = ref_frame.shape
            ref_frame = cv2.resize(ref_frame, (W // self.scale, H // self.scale), interpolation=cv2.INTER_LINEAR)
            # correct the size during inference time
            if self.use_full_frame:
                H, W, _ = input_frames.shape
                if H % self.scale != 0:
                    H = H // self.scale * self.scale
                    input_frames = input_frames[0:H, :, :]
                if W % self.scale != 0:
                    W = W // self.scale * self.scale
                    input_frames = input_frames[:, 0:W, :]
        
        # from numpy to torch.tensor
        input_frames = np.transpose(input_frames, (2, 0, 1))
        ref_frame = np.transpose(ref_frame, (2, 0, 1))

        input_frames = torch.from_numpy(input_frames).float() / 255.0
        ref_frame = torch.from_numpy(ref_frame).float() / 255.0

        return {'input_frames': input_frames, 'ref_frame': ref_frame, 'vis_flag': item_flag, 'scene': scene, 'entry': entry}

    def __len__(self):
        return len(self.items)
