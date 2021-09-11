import os
import shutil
import zipfile
import tarfile
import adobe_util

os.chdir('../..')
root = os.getcwd()
data_root_dir = os.path.join(root, 'datasets/Adobe-Matting')

fg_path = os.path.join(data_root_dir, 'train/fg')
a_path = os.path.join(data_root_dir, 'train/mask')
bg_path = os.path.join(data_root_dir, 'train/bg')
out_path = os.path.join(data_root_dir, 'train/merged')
fgdot_path = os.path.join(data_root_dir, 'train/fgdot')
bgcrop_path = os.path.join(data_root_dir, 'train/bgcrop')

train_folder = os.path.join(data_root_dir, 'Combined_Dataset/Training_set')

if not os.path.exists(os.path.join(data_root_dir,'Combined_Dataset')):
    zip_file = os.path.join(data_root_dir, 'Adobe_Deep_Matting_Dataset.zip')
    print('Extracting {}...'.format(zip_file))

    zip_ref = zipfile.ZipFile(zip_file, 'r')
    zip_ref.extractall(data_root_dir)
    zip_ref.close()

    if not os.path.exists(bg_path):
        zip_file = os.path.join(data_root_dir, 'train2014.zip')
        print('Extracting {}...'.format(zip_file))

        zip_ref = zipfile.ZipFile(zip_file, 'r')
        zip_ref.extractall(data_root_dir)
        zip_ref.close()

        with open(os.path.join(train_folder, 'training_bg_names.txt')) as f:
            training_bg_names = f.read().splitlines()

        os.makedirs(bg_path)
        for bg_name in training_bg_names:
            src_path = os.path.join(data_root_dir, 'train2014', bg_name)
            dest_path = os.path.join(bg_path, bg_name)
            shutil.move(src_path, dest_path)

if not os.path.exists(fg_path):
    os.makedirs(fg_path)

for old_folder in [train_folder + '/Adobe-licensed images/fg', train_folder + '/Other/fg']:
    fg_files = os.listdir(old_folder)
    for fg_file in fg_files:
        src_path = os.path.join(old_folder, fg_file)
        dest_path = os.path.join(fg_path, fg_file)
        shutil.move(src_path, dest_path)

if not os.path.exists(a_path):
    os.makedirs(a_path)

for old_folder in [train_folder + '/Adobe-licensed images/alpha', train_folder + '/Other/alpha']:
    a_files = os.listdir(old_folder)
    for a_file in a_files:
        src_path = os.path.join(old_folder, a_file)
        dest_path = os.path.join(a_path, a_file)
        shutil.move(src_path, dest_path)

if not os.path.exists(out_path):
    os.makedirs(out_path)
if not os.path.exists(fgdot_path):
    os.makedirs(fgdot_path)
if not os.path.exists(bgcrop_path):
    os.makedirs(bgcrop_path)

adobe_util.do_composite(fg_path, a_path, bg_path, out_path, fgdot_path, bgcrop_path, train_folder)

fg_test_path = os.path.join(data_root_dir, 'test/fg')
a_test_path = os.path.join(data_root_dir, 'test/mask')
bg_test_path = os.path.join(data_root_dir, 'test/bg')
out_test_path = os.path.join(data_root_dir, 'test/merged')
fgdot_test_path = os.path.join(data_root_dir, 'test/fgdot')
bgcrop_test_path = os.path.join(data_root_dir, 'test/bgcrop')

test_folder = os.path.join(data_root_dir, 'Combined_Dataset/Test_set')

if not os.path.exists(bg_test_path):
    os.makedirs(bg_test_path)

tar_file = os.path.join(data_root_dir, 'VOCtrainval_14-Jul-2008.tar')
print('Extracting {}...'.format(tar_file))

tar = tarfile.open(tar_file)
tar.extractall(data_root_dir)
tar.close()

tar_file = os.path.join(data_root_dir, 'VOC2008test.tar')
print('Extracting {}...'.format(tar_file))

tar = tarfile.open(tar_file)
tar.extractall(data_root_dir)
tar.close()

with open(os.path.join(test_folder, 'test_bg_names.txt')) as f:
    test_bg_names = f.read().splitlines()

for bg_name in test_bg_names:
    tokens = bg_name.split('_')
    src_path = os.path.join(os.path.join(data_root_dir, 'VOCdevkit/VOC2008/JPEGImages', bg_name))
    dest_path = os.path.join(bg_test_path, bg_name)
    shutil.move(src_path, dest_path)

if not os.path.exists(fg_test_path):
    os.makedirs(fg_test_path)

for old_folder in [test_folder + '/Adobe-licensed images/fg']:
    fg_files = os.listdir(old_folder)
    for fg_file in fg_files:
        src_path = os.path.join(old_folder, fg_file)
        dest_path = os.path.join(fg_test_path, fg_file)
        shutil.move(src_path, dest_path)

if not os.path.exists(a_test_path):
    os.makedirs(a_test_path)

for old_folder in [test_folder + '/Adobe-licensed images/alpha']:
    a_files = os.listdir(old_folder)
    for a_file in a_files:
        src_path = os.path.join(old_folder, a_file)
        dest_path = os.path.join(a_test_path, a_file)
        shutil.move(src_path, dest_path)

if not os.path.exists(out_test_path):
    os.makedirs(out_test_path)
if not os.path.exists(fgdot_test_path):
    os.makedirs(fgdot_test_path)
if not os.path.exists(bgcrop_test_path):
    os.makedirs(bgcrop_test_path)

adobe_util.do_composite_test(fg_test_path, a_test_path, bg_test_path, out_test_path, fgdot_test_path, bgcrop_test_path, test_folder)