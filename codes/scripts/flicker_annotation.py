import os
import json
import imagesize

def get_set_dict(data_root_dir, index_list, num_of_frames):
    imgs = sorted(os.listdir(data_root_dir))
    alls = [imgs[i] for i in index_list]

    num = len(alls)
    num_of_scenes = num // num_of_frames
    
    j = 0
    set_dict = {}
    for i in range(num_of_scenes):
        set_dict['scene_%04d' % i] = {}
        entry = 0
        set_dict['scene_%04d' % i]['%02d' % entry] = {}
        set_dict['scene_%04d' % i]['%02d' % entry]['ref_frame_path'] = os.path.join(data_root_dir, alls[j])
        for index in range(num_of_frames):
            set_dict['scene_%04d' % i]['%02d' % entry]['input_frame_%02d_path' % index] = os.path.join(data_root_dir, alls[j])
            j += 1
    return set_dict

os.chdir('../..')
root = os.getcwd()

data_root_dir = os.path.join(root, 'datasets/flicker/flicker_2W_images')
name = 'flicker'

annotation_dir = os.path.join(root, 'annotation')
if not os.path.exists(annotation_dir):
    os.makedirs(annotation_dir)

# get filtered index list
index_list = []
count = 0
for index, img in enumerate(sorted(os.listdir(data_root_dir))):
    if (index + 1) % 1000 == 0:
        print(index + 1, 'done!')
    img_path = os.path.join(data_root_dir, img)
    width, height = imagesize.get(img_path)
    if width < 576 or height < 576:
        pass
    else:
        count += 1
        index_list.append(index)

num_of_frames_list = [2, 3, 4, 5]
for num_of_frames in num_of_frames_list:
    # dict
    all_dict = get_set_dict(data_root_dir, index_list, num_of_frames)
    all_length = len(all_dict.keys())

    train_ratio = 0.7
    val_ratio = 0.1
    train_length = round(all_length * train_ratio)
    val_length = round(all_length * val_ratio)
    test_length = all_length - train_length - val_length

    # annotation
    train_path = os.path.join(annotation_dir, '%s_num%02d_train.json' % (name, num_of_frames))
    val_path = os.path.join(annotation_dir, '%s_num%02d_val.json' % (name, num_of_frames))
    test_path = os.path.join(annotation_dir, '%s_num%02d_test.json' % (name, num_of_frames))

    with open(train_path, 'w') as f:
        train_dict = dict(list(all_dict.items())[:train_length])
        json.dump(train_dict, f, sort_keys=True, indent=4)

    with open(val_path, 'w') as f:
        val_dict = dict(list(all_dict.items())[train_length:train_length+val_length])
        json.dump(val_dict, f, sort_keys=True, indent=4)

    with open(test_path, 'w') as f:
        test_dict = dict(list(all_dict.items())[-test_length:])
        json.dump(test_dict, f, sort_keys=True, indent=4)