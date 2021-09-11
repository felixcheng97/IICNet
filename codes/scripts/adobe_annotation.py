import os
import json

def get_set_dict(bg_dir, fg_dir, mask_dir, merged_dir, num=3):
    bgs = sorted(os.listdir(bg_dir), key=lambda x:x.lower())
    fgs = sorted(os.listdir(fg_dir), key=lambda x:x.lower())
    masks = sorted(os.listdir(mask_dir), key=lambda x:x.lower())
    mergeds = sorted(os.listdir(merged_dir), key=lambda x:(int(x.split('.')[0].split('_')[0]), int(x.split('.')[0].split('_')[1])))

    num_fg = len(fgs)
    num_bg = len(bgs)
    num_fg_per_bg = num_bg // num_fg

    j = 0
    set_dict = {}
    for i in range(num_fg):
        set_dict['scene_%03d' % i] = {}
        for entry in range(num_fg_per_bg):
            set_dict['scene_%03d' % i]['%02d' % entry] = {}
            set_dict['scene_%03d' % i]['%02d' % entry]['ref_frame_path'] = os.path.join(merged_dir, mergeds[j])
            set_dict['scene_%03d' % i]['%02d' % entry]['input_frame_00_path'] = os.path.join(merged_dir, mergeds[j])
            set_dict['scene_%03d' % i]['%02d' % entry]['input_frame_01_path'] = os.path.join(bg_dir, bgs[j])
            set_dict['scene_%03d' % i]['%02d' % entry]['input_frame_02_path'] = os.path.join(fg_dir, fgs[i])
            if num == 4:
                set_dict['scene_%03d' % i]['%02d' % entry]['input_frame_03_path'] = os.path.join(mask_dir, masks[i])
            j += 1
    return set_dict

data_root_dir = os.chdir('../..')
root = os.getcwd()

data_root_dir = os.path.join(root, 'datasets/Adobe-Matting')
name = 'adobe'

annotation_dir = os.path.join(root, 'annotation')
if not os.path.exists(annotation_dir):
    os.makedirs(annotation_dir)

# train val dict
bg_dir = os.path.join(data_root_dir, 'train/bgcrop')
fg_dir = os.path.join(data_root_dir, 'train/fgdot')
mask_dir = os.path.join(data_root_dir, 'train/mask')
merged_dir = os.path.join(data_root_dir, 'train/merged')
train_val_dict = get_set_dict(bg_dir, fg_dir, mask_dir, merged_dir)

all_length = len(train_val_dict)
train_dict = dict(list(train_val_dict.items())[:all_length-10])
val_dict = dict(list(train_val_dict.items())[-10:])

# test dict
bg_dir = os.path.join(data_root_dir, 'test/bgcrop')
fg_dir = os.path.join(data_root_dir, 'test/fgdot')
mask_dir = os.path.join(data_root_dir, 'test/mask')
merged_dir = os.path.join(data_root_dir, 'test/merged')
test_dict = get_set_dict(bg_dir, fg_dir, mask_dir, merged_dir)

# annotation
train_path = os.path.join(annotation_dir, '%s_train.json' % (name))
val_path = os.path.join(annotation_dir, '%s_val.json' % (name))
test_path = os.path.join(annotation_dir, '%s_test.json' % (name))

# write json files
with open(train_path, 'w') as f:
    json.dump(train_dict, f, sort_keys=True, indent=4)

with open(val_path, 'w') as f:
    json.dump(val_dict, f, sort_keys=True, indent=4)

with open(test_path, 'w') as f:
    json.dump(test_dict, f, sort_keys=True, indent=4)
