import os
import json

def get_set_dict(scene_dirs):
    set_dict = {}
    for scene_dir in scene_dirs:
        scene = scene_dir.split('/')[-1]
        set_dict[scene] = {}
        
        entry = 0
        frames = sorted(os.listdir(scene_dir))
        # each frame
        for frame in frames:
            frame_index = int(frame[:5])
            if frame_index + step * (num_of_frames - 1) < len(frames):
                set_dict[scene]['%02d' % entry] = {}
                for i in range(num_of_frames):
                    index = index_dict[i]
                    frame_i = '%05d.jpg' % (frame_index + step * i)
                    set_dict[scene]['%02d' % entry]['input_frame_%02d_path' % index] = os.path.join(scene_dir, frame_i)
                    if index % num_of_frames == 0:
                        set_dict[scene]['%02d' % entry]['ref_frame_path'] = os.path.join(scene_dir, frame_i)
                entry += 1
            else:
                break
    return set_dict

os.chdir('../..')
root = os.getcwd()

name = 'davis'
annotation_dir = os.path.join(root, 'annotation')
if not os.path.exists(annotation_dir):
    os.makedirs(annotation_dir)

num_of_frames_list = [3, 5, 7, 9]
step_list = [5, 3, 1]
pairs = [(num_of_frames, step) for num_of_frames in num_of_frames_list for step in step_list]

for num_of_frames, step in pairs:
    # create index dict
    index_dict = {**{i:i+1 for i in range(num_of_frames // 2)},
                **{num_of_frames // 2: 0},
                **{i:i for i in range(num_of_frames // 2 + 1, num_of_frames)}}

    # train set
    data_root_dir = os.path.join(root, 'datasets/DAVIS-2017/DAVIS-2017-trainval/JPEGImages/480p')
    train_scene_dirs = [os.path.join(data_root_dir, x) for x in sorted(os.listdir(data_root_dir))]

    # val set
    data_root_dir = os.path.join(root, 'datasets/DAVIS-2017/DAVIS-2017-test-dev/JPEGImages/480p')
    scene_dirs = [os.path.join(data_root_dir, x) for x in sorted(os.listdir(data_root_dir))]
    train_scene_dirs += scene_dirs[0::2]
    val_scene_dirs = scene_dirs[1::2]

    # test set
    data_root_dir = os.path.join(root, 'datasets/DAVIS-2017/DAVIS-2017-test-challenge/JPEGImages/480p')
    test_scene_dirs = [os.path.join(data_root_dir, x) for x in sorted(os.listdir(data_root_dir))]

    # get set dicts
    train_dict = get_set_dict(train_scene_dirs)
    val_dict = get_set_dict(val_scene_dirs)
    test_dict = get_set_dict(test_scene_dirs)

    # annotation
    train_path = os.path.join(annotation_dir, '%s_num%02d_step%02d_train.json' % (name, num_of_frames, step))
    val_path = os.path.join(annotation_dir, '%s_num%02d_step%02d_val.json' % (name, num_of_frames, step))
    test_path = os.path.join(annotation_dir, '%s_num%02d_step%02d_test.json' % (name, num_of_frames, step))

    with open(train_path, 'w') as f:
        json.dump(train_dict, f, sort_keys=True, indent=4)
    with open(val_path, 'w') as f:
        json.dump(val_dict, f, sort_keys=True, indent=4)
    with open(test_path, 'w') as f:
        json.dump(test_dict, f, sort_keys=True, indent=4)