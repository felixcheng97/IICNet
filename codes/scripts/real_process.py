#######################################
# Prepares training data. Takes a path to a directory of videos + captured backgrounds, dumps frames, extracts human
# segmentations. Also takes a path of background videos. Creates a training CSV file with lines of the following format,
# by using all but the last 80 frames of each video and iterating repeatedly over the background frames as needed.

#$image;$captured_back;$segmentation;$image+20frames;$image+2*20frames;$image+3*20frames;$image+4*20frames;$target_back

# path = "/path/to/Captured_Data/fixed-camera/train"

import os
from itertools import cycle
from tqdm import tqdm

os.chdir('../..')
root = os.getcwd()
data_root_dir = os.path.join(root, 'datasets/Real-Matting')
paths = []
for capture_method in os.listdir(data_root_dir):
    capture_method_dir = os.path.join(data_root_dir, capture_method)
    if not os.path.isdir(capture_method_dir):
        continue
    for folder in os.listdir(capture_method_dir):
        folder_dir = os.path.join(capture_method_dir, folder)
        if not os.path.isdir(folder_dir):
            continue
        paths.append(folder_dir)

for path in paths:
    videos = [os.path.join(path, f[:-4]) for f in os.listdir(path) if f.endswith(".mp4")]
    # backgrounds = [os.path.join(background_path, f[:-4]) for f in os.listdir(background_path) if f.endswith(".MOV")]

    print(f"Dumping frames of {len(videos)} input videos")
    for i, video in enumerate(tqdm(videos)):
        os.makedirs(video, exist_ok=True)
        code = os.system(f"ffmpeg -i {video}.mp4 {video}/%04d_img.png -hide_banner > real_process_logs.txt 2>&1")
        if code != 0:
            exit(code)
        print(f"Dumped frames for {video} ({i+1}/{len(videos)})")
    
# remove the log file if needed
os.remove("./real_process_logs.txt")
print("File Removed!")