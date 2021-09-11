import os
import math
import time
import cv2 as cv
import numpy as np
from tqdm import tqdm

def composite4(fg, bg, a, w, h):
    fg = np.array(fg, np.float32)
    bgcrop = np.array(bg[0:h, 0:w], np.float32)
    alpha = np.zeros((h, w, 1), np.float32)
    alpha[:, :, 0] = a / 255.
    comp = alpha * fg + (1 - alpha) * bgcrop
    comp = comp.astype(np.uint8)
    return comp, bgcrop


def process(fg_path, a_path, bg_path, out_path, fgdot_path, bgcrop_path, im_name, bg_name, fcount, bcount):
    im = cv.imread(os.path.join(fg_path, im_name))
    a = cv.imread(os.path.join(a_path, im_name), 0)
    h, w = im.shape[:2]
    bg = cv.imread(os.path.join(bg_path, bg_name))
    bh, bw = bg.shape[:2]
    wratio = w / bw
    hratio = h / bh
    ratio = wratio if wratio > hratio else hratio
    if ratio > 1:
        bg = cv.resize(src=bg, dsize=(math.ceil(bw * ratio), math.ceil(bh * ratio)), interpolation=cv.INTER_CUBIC)

    out, bgcrop = composite4(im, bg, a, w, h)
    filename = os.path.join(out_path, str(fcount) + '_' + str(bcount) + '.png')
    cv.imwrite(filename, out)

    filename = os.path.join(bgcrop_path, bg_name)
    cv.imwrite(filename, bgcrop)

    a_mask = a
    a_mask[a_mask > 0] = 1
    a_mask[a_mask == 0] = 0
    fgdot = im * np.expand_dims(a_mask, 2)
    filename = os.path.join(fgdot_path, im_name)
    cv.imwrite(filename, fgdot)


def do_composite_test(fg_path, a_path, bg_path, out_path, fgdot_path, bgcrop_path, folder):
    num_bgs = 20

    with open(os.path.join(folder, 'test_bg_names.txt')) as f:
        bg_files = f.read().splitlines()
    with open(os.path.join(folder, 'test_fg_names.txt')) as f:
        fg_files = f.read().splitlines()

    # a_files = os.listdir(a_path)
    num_samples = len(fg_files) * num_bgs

    # pb = ProgressBar(total=100, prefix='Compose test images', suffix='', decimals=3, length=50, fill='=')
    start = time.time()
    bcount = 0
    for fcount in tqdm(range(len(fg_files))):
        im_name = fg_files[fcount]

        for i in range(num_bgs):
            bg_name = bg_files[bcount]
            process(fg_path, a_path, bg_path, out_path, fgdot_path, bgcrop_path, im_name, bg_name, fcount, bcount)
            bcount += 1

            # pb.print_progress_bar(bcount * 100.0 / num_samples)

    end = time.time()
    elapsed = end - start
    print('elapsed: {} seconds'.format(elapsed))


def do_composite(fg_path, a_path, bg_path, out_path, fgdot_path, bgcrop_path, folder):
    num_bgs = 100

    with open(os.path.join(folder, 'training_bg_names.txt')) as f:
        bg_files = f.read().splitlines()
    with open(os.path.join(folder, 'training_fg_names.txt')) as f:
        fg_files = f.read().splitlines()

    # a_files = os.listdir(a_path)
    num_samples = len(fg_files) * num_bgs

    # pb = ProgressBar(total=100, prefix='Compose train images', suffix='', decimals=3, length=50, fill='=')
    start = time.time()
    bcount = 0
    for fcount in tqdm(range(len(fg_files))):
        im_name = fg_files[fcount]

        for i in range(num_bgs):
            bg_name = bg_files[bcount]
            process(fg_path, a_path, bg_path, out_path, fgdot_path, bgcrop_path, im_name, bg_name, fcount, bcount)
            bcount += 1

            # pb.print_progress_bar(bcount * 100.0 / num_samples)

    end = time.time()
    elapsed = end - start
    print('elapsed: {} seconds'.format(elapsed))