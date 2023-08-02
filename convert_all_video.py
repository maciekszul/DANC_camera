import math

from utilities import files
import numpy as np
import cv2
import time
import sys
import os.path as op
import os
import json
import shutil
import psutil
from utilities.tools import makefolder

from joblib import Parallel, delayed

def compute_cb_gamma_luts(img, percent=1):
    img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR)
    out_channels = []
    cumstops = (
        img.shape[0] * img.shape[1] * percent / 200.0,
        img.shape[0] * img.shape[1] * (1 - percent / 200.0)
    )
    cb_luts=[]
    for channel in cv2.split(img):
        cumhist = np.cumsum(cv2.calcHist([channel], [0], None, [256], (0,256)))
        low_cut, high_cut = np.searchsorted(cumhist, cumstops)
        lut = np.concatenate((
            np.zeros(low_cut),
            np.around(np.linspace(0, 255, high_cut - low_cut + 1)),
            255 * np.ones(255 - high_cut)
        ))
        cb_luts.append(lut)
        out_channels.append(cv2.LUT(channel, lut.astype('uint8')))
    img=cv2.merge(out_channels)


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mid=0.5
    mean=np.mean(gray)
    gamma=math.log(mid*255)/math.log(mean)
    gamma_lut = np.empty((1, 256), np.uint8)
    inv_gamma = 1.0 / gamma
    for i in range(256):
        gamma_lut[0, i] = np.clip(pow(i / 255.0, inv_gamma) * 255.0, 0, 255)

    return cb_luts, gamma_lut

def convert(path, file, json_file, out_path):
    filename = file.split("/")[-1].split(".")[0]
    raw = np.load(file, allow_pickle=True)

    with open(json_file) as file_j:
        metadata = json.load(file_j)

    cb_luts, gamma_lut = compute_cb_gamma_luts(raw[0], percent=5)

    f_size = (1280, 1024)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    vid = cv2.VideoWriter(
        "{}/{}.avi".format(out_path, filename),
        fourcc,
        float(200),
        f_size
    )
    for i in raw:
        img = cv2.cvtColor(i, cv2.COLOR_BAYER_BG2BGR)

        # Color balance
        out_channels = []
        for channel, lut in zip(cv2.split(img), cb_luts):
            out_channels.append(cv2.LUT(channel, lut.astype('uint8')))
        img=cv2.merge(out_channels)

        # Gamma correction
        img = cv2.LUT(img, gamma_lut)

        vid.write(img)
    
    vid.release()
    print(filename, "saved")
    os.remove(file)
    shutil.move(json_file, op.join(out_path, json_file.split('/')[-1]))


if __name__=='__main__':
    # opening a json file
    with open('settings.json') as settings_file:
        params = json.load(settings_file)
    assert(psutil.disk_usage(params['output_dir1']).percent<75)

    recording_dirs = files.get_folders_files(params['output_dir1'])[0]
    recording_dirs.extend(files.get_folders_files(params['output_dir2'])[0])

    for recording_dir in recording_dirs:
        print(recording_dir)
        sub_dir = files.get_folders(recording_dir, 'sub-')[0]
        blk_dirs = files.get_folders(op.join(recording_dir, sub_dir), 'block_')
        for blk_dir in blk_dirs:
            files_npy = files.get_files(op.join(recording_dir, sub_dir, blk_dir), "", ".npy")[2]
            files_npy.sort()
            files_json = files.get_files(op.join(recording_dir, sub_dir, blk_dir), "", ".json")[2]
            files_json.sort()

            files_npy_json = list(zip(files_npy, files_json))

            pth=op.join(recording_dir, sub_dir, blk_dir)
            makefolder(op.join(params['save_dir'], op.split(recording_dir)[-1]))
            makefolder(op.join(params['save_dir'], op.split(recording_dir)[-1], sub_dir))
            out_path=op.join(params['save_dir'], op.split(recording_dir)[-1], sub_dir, blk_dir)
            makefolder(out_path)
            Parallel(n_jobs=-1)(
                delayed(convert)(pth, file, json_file, out_path) for file, json_file in files_npy_json)
        shutil.rmtree(recording_dir)
        #shutil.move(recording_dir, params['save_dir'])

