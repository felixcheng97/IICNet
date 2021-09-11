import os
import json
import imagesize

def get_set_dict(hr_dir, lr_dir, check):
    hrs = sorted(os.listdir(hr_dir))
    lrs = sorted(os.listdir(lr_dir))

    num = len(hrs)

    set_dict = {}
    for i in range(num):
        # check resolution for X8:
        if check:
            width, height = imagesize.get(os.path.join(lr_dir, lrs[i]))
            if width < 144 or height < 144:
                continue
        set_dict['scene_%03d' % (i + 1)] = {}
        entry = 0
        set_dict['scene_%03d' % (i + 1)]['%02d' % entry] = {}
        set_dict['scene_%03d' % (i + 1)]['%02d' % entry]['ref_frame_path'] = os.path.join(lr_dir, lrs[i])
        set_dict['scene_%03d' % (i + 1)]['%02d' % entry]['input_frame_00_path'] = os.path.join(lr_dir, lrs[i])
        set_dict['scene_%03d' % (i + 1)]['%02d' % entry]['input_frame_01_path'] = os.path.join(hr_dir, hrs[i])
    return set_dict

os.chdir('../..')
root = os.getcwd()

data_root_dir = os.path.join(root, 'datasets/DIV2K')
name = 'div2kdual'

annotation_dir = os.path.join(root, 'annotation')
if not os.path.exists(annotation_dir):
    os.makedirs(annotation_dir)

zoom_list = [2, 4, 8]
for zoom in zoom_list:
    check = True if zoom == 8 else False

    # dict
    hr_dir = os.path.join(data_root_dir, 'DIV2K_train_HR')
    lr_dir = os.path.join(data_root_dir, 'DIV2K_train_LR_bicubic/X%d' % zoom)
    train_val_dict = get_set_dict(hr_dir, lr_dir, check)

    train_val_length = len(train_val_dict)
    train_dict = dict(list(train_val_dict.items())[:train_val_length-50])
    val_dict = dict(list(train_val_dict.items())[-50:])

    hr_dir = os.path.join(data_root_dir, 'DIV2K_valid_HR')
    lr_dir = os.path.join(data_root_dir, 'DIV2K_valid_LR_bicubic/X%d' % zoom)
    test_dict = get_set_dict(hr_dir, lr_dir, False)

    # annotation
    train_path = os.path.join(annotation_dir, '%s_X%d_train.json' % (name, zoom))
    val_path = os.path.join(annotation_dir, '%s_X%d_val.json' % (name, zoom))
    test_path = os.path.join(annotation_dir, '%s_X%d_test.json' % (name, zoom))

    # write json files
    with open(train_path, 'w') as f:
        json.dump(train_dict, f, sort_keys=True, indent=4)

    with open(val_path, 'w') as f:
        json.dump(val_dict, f, sort_keys=True, indent=4)

    with open(test_path, 'w') as f:
        json.dump(test_dict, f, sort_keys=True, indent=4)
