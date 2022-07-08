import os
import socket
import threading
import time
import json
from datetime import datetime

import numpy as np
import os.path as op

from bluepy import btle
from joblib import Parallel, delayed

from convert_all_video import convert
from sensor import GyroAccelSensor
import pandas as pd

from camera_io import init_camera_sources, shtr_spd
from utilities import files
from utilities.tools import makefolder, dump_the_dict

fps = 200
gain = 5
shutter = shtr_spd(fps)

buffer_size = 60

IO_SAMP_CHAR_UUID = "6a80ff0c-b5a3-f393-e0a9-e50e24dcca9e"

def dump_and_run(lists, path):
    frames = np.array(lists)
    np.save(path, frames)

class accThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.sdata = None
        self.acc_fname = None
        self.recording = False
        self.connected = False
        self.shutdown_flag = False

    def start_recording(self, subject, block, trial, timestamp):
        subj_dir = op.join(out_dir, 'sub_{}'.format(subject))
        blk_dir = op.join(subj_dir, 'block_{}'.format(block))
        self.acc_fname = os.path.join(blk_dir,
                                 "block-{}_trial-{}_{}_acc-data.csv".format(block,
                                                                      str(trial).zfill(3),
                                                                      timestamp)
                                 )
        self.sdata = pd.DataFrame()
        self.block = block
        self.trial = trial
        self.recording = True

    def stop_recording(self):
        self.recording = False
        self.sdata.to_csv(self.acc_fname)
        self.sdata = None
        self.acc_fname = None

    def shutdown(self):
        self.shutdown_flag = True

    def run(self):
        print('Scanning')
        scanner = btle.Scanner()
        address = None
        while address is None:
           devices = scanner.scan(10.0)
           for dev in devices:
               addr = dev.addr
               name = dev.getValueText(9)
               if name == 'SENSOR_PRO':
                   print('%s: %s' % (name, addr))
                   address = addr
                   break

        print('Connecting')
        self.dev = btle.Peripheral(address, "random")

        MPU6050Service = self.dev.getServiceByUUID('6a800001-b5a3-f393-e0a9-e50e24dcca9e')
        SampIntChar = MPU6050Service.getCharacteristics('6a80ff0c-b5a3-f393-e0a9-e50e24dcca9e')[0]
        self.dev.writeCharacteristic(SampIntChar.valHandle, b"\x00\x64")

        self.gyro_acc = GyroAccelSensor(self.dev)
        self.gyro_acc.enable()

        self.connected = True
        try:
            while True:
               (gyro, accel) = self.gyro_acc.read()
               if self.shutdown_flag:
                   break

               if self.recording:
                   self.sdata = self.sdata.append(
                            {
                               'gyro_x': gyro[0, 0],
                               'gyro_y': gyro[0, 1],
                               'gyro_z': gyro[0, 2],
                               'accel_x': accel[0, 0],
                               'accel_y': accel[0, 1],
                               'accel_z': accel[0, 2],
                               'block': str(self.block),
                               'trial': str(self.trial)
                            }, ignore_index=True)
        finally:
            self.gyro_acc.disable()
            self.dev.disconnect()

# opening a json file
with open('settings.json') as settings_file:
    params = json.load(settings_file)

if __name__=='__main__':

    thread1 = accThread()
    thread1.start()

    cams = init_camera_sources(params, fps, shutter, gain, sensor_feature_value=1, disable_auto_bandwidth=True,
                               img_data_format='XI_RAW8', auto_wb=False,
                               counter_selector='XI_CNT_SEL_API_SKIPPED_FRAMES')

    makefolder('./data')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir=os.path.join('./data',timestamp)
    os.mkdir(out_dir)

    settings_file = os.path.join(out_dir, 'settings.json')
    dump_the_dict(settings_file, params)

    # Wait for accelerometer to connect
    while not thread1.connected:
        time.sleep(10)

    print('Ready')

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((params['TCP_IP'], params['TCP_PORT']))
    message_connect = "connected"
    s.send(message_connect.encode())

    block = -1

    while True:
        try:
            data_raw = s.recv(buffer_size)
            data = data_raw.decode()
        except socket.error:
            data=''

        if "start" in data:
            subject, block, trial, status, timestamp = data.split("_")
            print(subject, "-", block, "-", trial, ": start")

            thread1.start_recording(subject, block, trial, timestamp)

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

            subj_dir = op.join(out_dir, 'sub_{}'.format(subject))
            makefolder(subj_dir)
            blk_dir = op.join(subj_dir, 'block_{}'.format(block))
            makefolder(blk_dir)

            thread1.stop_recording()

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
            for cam in cams:
                cam.stop()

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

            for cam in cams:
                cam.start()

        if "exit" in data:
            break

    for cam in cams:
        cam.close()

    s.close()
    thread1.shutdown()




