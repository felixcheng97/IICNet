import os
import json

def get_dir_dict(data_root_dir):
    dir_dict = {}
    entry = 0
    images = sorted(os.listdir(data_root_dir))
    left_images = images[0::2]
    right_images = images[1::2]
    for left_image, right_image in zip(left_images, right_images):
        scene = left_image.split('_')[0]
        dir_dict[scene] = {}
        dir_dict[scene]['%02d' % entry] = {}
        dir_dict[scene]['%02d' % entry]['input_frame_00_path'] = os.path.join(data_root_dir, left_image)
        dir_dict[scene]['%02d' % entry]['input_frame_01_path'] = os.path.join(data_root_dir, right_image)
        dir_dict[scene]['%02d' % entry]['ref_frame_path'] = os.path.join(data_root_dir, left_image)
    return dir_dict

os.chdir('../..')
root = os.getcwd()

name = 'flicker1024'
annotation_dir = os.path.join(root, 'annotation')
if not os.path.exists(annotation_dir):
    os.makedirs(annotation_dir)

# train set
train_list = []
data_root_dirs = [os.path.join(root, 'datasets/flicker1024/Train_1'),
                  os.path.join(root, 'datasets/flicker1024/Train_2'),
                  os.path.join(root, 'datasets/flicker1024/Train_3'),
                  os.path.join(root, 'datasets/flicker1024/Train_4')]
for data_root_dir in data_root_dirs:
    train_list += list(get_dir_dict(data_root_dir).items())
train_dict = dict(train_list)

# val set
data_root_dir =  os.path.join(root, 'datasets/flicker1024/Validation')
val_dict = get_dir_dict(data_root_dir)

# test set
data_root_dir = os.path.join(root, 'datasets/flicker1024/Test')
test_dict = get_dir_dict(data_root_dir)

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
