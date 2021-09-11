import os
import json
import imagesize

def get_set_dict(data_root_dir, index_list):
    imgs = sorted(os.listdir(data_root_dir))
    alls = [imgs[i] for i in index_list]
    num = len(alls)
    
    set_dict = {}
    for i in range(num):
        set_dict['scene_%05d' % i] = {}
        entry = 0
        set_dict['scene_%05d' % i]['%02d' % entry] = {}
        set_dict['scene_%05d' % i]['%02d' % entry]['img_path'] = os.path.join(data_root_dir, alls[i])
    return set_dict

os.chdir('../..')
root = os.getcwd()

data_root_dir = os.path.join(root, 'datasets/VOCdevkit/VOC2012/JPEGImages')
name = 'voc2012'

annotation_dir = os.path.join(root, 'annotation')
if not os.path.exists(annotation_dir):
    os.makedirs(annotation_dir)

index_list = []
count = 0
filtered = 0
for index, img in enumerate(sorted(os.listdir(data_root_dir))):
    if (index + 1) % 1000 == 0:
        print(index + 1, 'done!')
    img_path = os.path.join(data_root_dir, img)
    width, height = imagesize.get(img_path)
    if width < 144 or height < 144:
        pass
    else:
        count += 1
        index_list.append(index)

# dict
all_dict = get_set_dict(data_root_dir, index_list)
all_length = len(all_dict.keys())

train_length = 13758
test_length = all_length - train_length

# annotation
train_path = os.path.join(annotation_dir, '%s_train.json' % (name))
test_path = os.path.join(annotation_dir, '%s_test.json' % (name))

with open(train_path, 'w') as f:
    train_dict = dict(list(all_dict.items())[:train_length])
    json.dump(train_dict, f, sort_keys=True, indent=4)

with open(test_path, 'w') as f:
    test_dict = dict(list(all_dict.items())[-test_length:])
    json.dump(test_dict, f, sort_keys=True, indent=4)
