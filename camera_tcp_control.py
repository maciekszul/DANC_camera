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

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((params['TCP_IP'], params['TCP_PORT']))
message_connect = "connected"
s.send(message_connect.encode())

print('Ready')

makefolder('./data')
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
out_dir=os.path.join('./data',timestamp)
os.mkdir(out_dir)

settings_file=os.path.join(out_dir, 'settings.json')
dump_the_dict(settings_file, params)

block=-1

while True:
    try:
        data_raw = s.recv(buffer_size)
        data = data_raw.decode()
    except socket.error:
        data=''

    if "start" in data:
        subject, block, trial, status, timestamp = data.split("_")
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
        s.setblocking(0)
        while True:
            try:
                data_raw = s.recv(buffer_size)
                data = data_raw.decode()
            except socket.error:
                data=''


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
            if "stop" in data:
                s.setblocking(1)
                break

        stop = time.monotonic()
        print("recorded_in", stop - start)

        sub_dir = op.join(out_dir, 'sub_{}'.format(subject))
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
        message_dump = "dumped_{}_rec_{}".format(dump_time, total_rec)
        s.send(message_dump.encode())
    if "convert" in data:
        print(blk_dir)
        start_x = time.monotonic()

        files_npy = files.get_files(blk_dir, "", ".npy")[2]
        files_npy.sort()
        files_json = files.get_files(blk_dir, "", ".json")[2]
        files_json.sort()

        files_npy_json = list(zip(files_npy, files_json))

        Parallel(n_jobs=-1)(delayed(convert)(blk_dir, file, json_file) for file, json_file in files_npy_json)

        stop_x = time.monotonic()
        convert_time = stop_x - start_x
        print("DATA CONVERTED IN:", convert_time)
        message_convert = "converted_{}".format(convert_time)
        s.send(message_convert.encode())

    if "exit" in data:
        break

for cam in cams:
    cam.close()

s.close()
