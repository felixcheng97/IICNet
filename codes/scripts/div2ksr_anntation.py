import os
import json

def get_set_dict(hr_dir, lr_dir):
    hrs = sorted(os.listdir(hr_dir))
    lrs = sorted(os.listdir(lr_dir))
    
    num = len(hrs)

    set_dict = {}
    for i in range(num):
        set_dict['scene_%03d' % (i + 1)] = {}
        entry = 0
        set_dict['scene_%03d' % (i + 1)]['%02d' % entry] = {}
        set_dict['scene_%03d' % (i + 1)]['%02d' % entry]['ref_frame_path'] = os.path.join(lr_dir, lrs[i])
        set_dict['scene_%03d' % (i + 1)]['%02d' % entry]['input_frame_00_path'] = os.path.join(hr_dir, hrs[i])
    return set_dict

os.chdir('../..')
root = os.getcwd()

data_root_dir = os.path.join(root, 'datasets/DIV2K')
name = 'div2ksr'
zoom = 2

annotation_dir = os.path.join(root, 'annotation')
if not os.path.exists(annotation_dir):
    os.makedirs(annotation_dir)

# dict
hr_dir = os.path.join(data_root_dir, 'DIV2K_train_HR')
lr_dir = os.path.join(data_root_dir, 'DIV2K_train_LR_bicubic/X%d' % zoom)
train_dict = get_set_dict(hr_dir, lr_dir)

hr_dir = os.path.join(data_root_dir, 'DIV2K_valid_HR')
lr_dir = os.path.join(data_root_dir, 'DIV2K_valid_LR_bicubic/X%d' % zoom)
test_dict = get_set_dict(hr_dir, lr_dir)

# annotation
train_path = os.path.join(annotation_dir, '%s_X%d_train.json' % (name, zoom))
test_path = os.path.join(annotation_dir, '%s_X%d_test.json' % (name, zoom))

# write json files
with open(train_path, 'w') as f:
    json.dump(train_dict, f, sort_keys=True, indent=4)

with open(test_path, 'w') as f:
    json.dump(test_dict, f, sort_keys=True, indent=4)
