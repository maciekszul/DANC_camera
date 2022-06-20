import math

from utilities import files
import numpy as np
import cv2
import time
import sys
import os.path as op
import os
import json
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

def convert(path, file, json_file):
    filename = file.split("/")[-1].split(".")[0]
    raw = np.load(file, allow_pickle=True)

    with open(json_file) as file_j:
        metadata = json.load(file_j)

    cb_luts, gamma_lut = compute_cb_gamma_luts(raw[0], percent=5)

    f_size = (1280, 1024)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    vid = cv2.VideoWriter(
        "{}/{}.avi".format(path, filename),
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


if __name__=='main':
    try:
        path = str(sys.argv[1])
    except:
        print("incorrect path")
        sys.exit()

    print(path)

    files_npy = files.get_files(path, "", ".npy")[2]
    files_npy.sort()
    files_json = files.get_files(path, "", ".json")[2]
    files_json.sort()

    files_npy_json = list(zip(files_npy, files_json))

    # print(files_npy_json)
    Parallel(n_jobs=-1)(delayed(convert)(path, file, json_file) for file, json_file in files_npy_json)