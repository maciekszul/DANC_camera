import os
import socket
import time
import json
from datetime import datetime

import numpy as np
import os.path as op

from camera_io import init_camera_sources, shtr_spd
from convert_all_video import convert
from utilities import files
from utilities.tools import makefolder, dump_the_dict
import psutil

from joblib import Parallel, delayed

def dump_and_run(lists, path):
    frames = np.array(lists)
    np.save(path, frames)


# opening a json file
with open('settings.json') as settings_file:
    params = json.load(settings_file)

fps = 200
gain = 5
shutter = shtr_spd(fps)

cams=init_camera_sources(params, fps, shutter, gain, sensor_feature_value=1, disable_auto_bandwidth=True,
                         img_data_format='XI_RAW8', auto_wb=False, counter_selector='XI_CNT_SEL_API_SKIPPED_FRAMES')


buffer_size = 60

print('Ready')

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
out_dir=os.path.join(params['output_dir1'],timestamp)
if psutil.disk_usage(params['output_dir1']).percent>=73:
    out_dir=os.path.join(params['output_dir2'],timestamp)
os.mkdir(out_dir)

settings_file=os.path.join(out_dir, 'settings.json')
dump_the_dict(settings_file, params)

block=-1

for tr_idx in range(250):
    
    subject='test'
    block='0'
    trial=str(tr_idx)
    
    print(subject, "-", block, "-", trial, ": start")

    metadata_cam0 = {
        "block": block,
        "trial": trial,
        "frame_timestamp": [],
        "framerate": fps,
        "shutter_speed": shutter,
        "gain": gain,
        "sn": params['cam_sns'][0]
    }

    metadata_cam1 = {
        "block": block,
        "trial": trial,
        "frame_timestamp": [],
        "framerate": fps,
        "shutter_speed": shutter,
        "gain": gain,
        "sn": params['cam_sns'][1]
    }

    metadata_cam2 = {
        "block": block,
        "trial": trial,
        "frame_timestamp": [],
        "framerate": fps,
        "shutter_speed": shutter,
        "gain": gain,
        "sn": params['cam_sns'][2]
    }

    metadata_cam3 = {
        "block": block,
        "trial": trial,
        "frame_timestamp": [],
        "framerate": fps,
        "shutter_speed": shutter,
        "gain": gain,
        "sn": params['cam_sns'][3]
    }

    cam0_l = []
    cam1_l = []
    cam2_l = []
    cam3_l = []

    start = time.monotonic()
    
    for fr in range(1000):
        

        co0 = cams[0].next_frame()
        cam0_l.append(co0)
        metadata_cam0["frame_timestamp"].append(time.monotonic())

        co1 = cams[1].next_frame()
        cam1_l.append(co1)
        metadata_cam1["frame_timestamp"].append(time.monotonic())

        co2 = cams[2].next_frame()
        cam2_l.append(co2)
        metadata_cam2["frame_timestamp"].append(time.monotonic())

        co3 = cams[3].next_frame()
        cam3_l.append(co3)
        metadata_cam3["frame_timestamp"].append(time.monotonic())


        # print(counter)
    
    stop = time.monotonic()
    print("recorded_in", stop - start)

    if psutil.disk_usage(params['output_dir1']).percent>=73:
        out_dir=os.path.join(params['output_dir2'],timestamp)
        makefolder(out_dir)
        
    sub_dir = op.join(out_dir, subject)
    makefolder(sub_dir)
    blk_dir = op.join(sub_dir,'block_{}'.format(block))
    makefolder(blk_dir)

    start_x = time.monotonic()
    total_rec = start_x - start
    for ix, v in enumerate([cam0_l, cam1_l, cam2_l, cam3_l]):
        filename = "block-{}_trial-{}_cam-{}_frames-{}_{}".format(
            block,
            trial,
            params['cam_sns'][ix],
            str(len(v)).zfill(4),
            timestamp
        )
        npy_path = op.join(blk_dir, filename + ".npy")
        json_path = op.join(blk_dir, filename + ".json")
        dump_and_run(v, npy_path)
        with open(json_path, "w") as fp:
            json.dump([metadata_cam0, metadata_cam1, metadata_cam2, metadata_cam3][ix], fp)

    stop_x = time.monotonic()
    dump_time = stop_x - start_x
    print("DATA DUMPED IN:", dump_time)

for cam in cams:
    cam.close()

