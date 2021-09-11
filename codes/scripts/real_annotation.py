import os
import json
import pandas as pd
from shutil import copyfile

def get_set_dict(data_root_dir, label):
    set_dict = {}
    for scene, row in label.iterrows():
        mode = row['mode']
        folder = row['folder']
        filename = row['filename']
        background_id = row['background_id']
        set_dir = os.path.join(data_root_dir, mode, folder, filename)
        set_dict['scene_%02d' % scene] = {}
        for entry, frame in enumerate(sorted(os.listdir(set_dir))):
            set_dict['scene_%02d' % scene]['%04d' % entry] = {}
            set_dict['scene_%02d' % scene]['%04d' % entry]['input_frame_00_path'] = os.path.join(set_dir, frame)
            set_dict['scene_%02d' % scene]['%04d' % entry]['input_frame_01_path'] = set_dir + '.png'
            set_dict['scene_%02d' % scene]['%04d' % entry]['ref_frame_path'] = os.path.join(set_dir, frame)
    return set_dict

os.chdir('../..')
root = os.getcwd()

data_root_dir = os.path.join(root, 'datasets/Real-Matting')
name = 'real'

annotation_dir = os.path.join(root, 'annotation')
if not os.path.exists(annotation_dir):
    os.makedirs(annotation_dir)

label = pd.read_csv('./codes/scripts/label.csv')
label = label[['mode', 'folder', 'filename', 'background_id']]
train_label = label[(label['background_id'] != 1) & (label['background_id'] != 3) & (label['background_id'] != 10) & (label['background_id'] != 12)]
val_label = label[(label['background_id'] == 12)]
test_label = label[(label['background_id'] == 1) | (label['background_id'] == 3) | (label['background_id'] == 10)]

train_dict = get_set_dict(data_root_dir, train_label)
val_dict = get_set_dict(data_root_dir, val_label)
test_dict = get_set_dict(data_root_dir, test_label)

# annotation
train_path = os.path.join(annotation_dir, '%s_train.json' % (name))
val_path = os.path.join(annotation_dir, '%s_val.json' % (name))
test_path = os.path.join(annotation_dir, '%s_test.json' % (name))

with open(train_path, 'w') as f:
    json.dump(train_dict, f, sort_keys=True, indent=4)

with open(val_path, 'w') as f:
    json.dump(val_dict, f, sort_keys=True, indent=4)

with open(test_path, 'w') as f:
    json.dump(test_dict, f, sort_keys=True, indent=4)
