from utilities import files
import numpy as np
import cv2
import time
import sys
import os.path as op
import os
import json
from joblib import Parallel, delayed


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

def convert(file, json_file):
    filename = file.split("/")[-1].split(".")[0]
    raw = np.load(file, allow_pickle=True)

    with open(json_file) as file_j:
        metadata = json.load(file_j)

    Rwb, Gwb, Bwb = metadata["WB_auto_coeff_RGB"]

    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, 0.5) * 255.0, 0, 255)

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

        img = cv2.normalize(img, np.zeros(img.shape[1:]), 0, 255, cv2.NORM_MINMAX)

        img = cv2.LUT(img, lookUpTable)

        img = cv2.convertScaleAbs(img, alpha=0.7)

        thr = 235

        img[:,:,0][img[:,:,0] < thr] = img[:,:,0][img[:,:,0] < thr] * Bwb
        img[:,:,1][img[:,:,1] < thr] = img[:,:,1][img[:,:,1] < thr] * Gwb
        img[:,:,2][img[:,:,2] < thr] = img[:,:,2][img[:,:,2] < thr] * Rwb

        vid.write(img)
    
    vid.release()
    print(filename, "saved")
    os.remove(file)

Parallel(n_jobs=-1)(delayed(convert)(file, json_file) for file, json_file in files_npy_json)
