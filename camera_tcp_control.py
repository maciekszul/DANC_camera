import socket
import time
import json
import numpy as np
import os.path as op
from ximea import xiapi
from datetime import datetime
from copy import copy


def get_WB_coef(s_n, framerate, shutter, gain):
    cam = xiapi.Camera()
    img = xiapi.Image()
    cam.open_device_by_SN(s_n)
    cam.set_sensor_feature_value(1)
    cam.set_imgdataformat("XI_RGB24")
    cam.set_acq_timing_mode("XI_ACQ_TIMING_MODE_FRAME_RATE")
    cam.set_framerate(framerate)
    cam.set_exposure(shutter)
    cam.set_gain(gain)
    cam.enable_auto_wb()
    cam.start_acquisition()
    start = time.monotonic()
    while (time.monotonic() - start) <= 1:
        cam.get_image(img)

    kR = cam.get_wb_kr()
    kG = cam.get_wb_kg()
    kB = cam.get_wb_kb()

    cam.stop_acquisition()
    cam.close_device()

    return kR, kG, kB


def camera_init(s_n, framerate, shutter, gain):
    cam = xiapi.Camera()
    img = xiapi.Image()
    cam.open_device_by_SN(s_n)
    cam.set_sensor_feature_value(1)
    cam.set_imgdataformat("XI_RAW8")
    cam.disable_auto_bandwidth_calculation()
    cam.disable_auto_wb()
    cam.set_counter_selector("XI_CNT_SEL_API_SKIPPED_FRAMES")
    cam.set_acq_timing_mode("XI_ACQ_TIMING_MODE_FRAME_RATE")
    cam.set_framerate(framerate)
    cam.set_exposure(shutter)
    cam.set_gain(gain)
    return cam, img



def shtr_spd(framerate):
    return int((1/framerate)*1e+6)-100


def dump_and_run(lists, path):
    frames = np.array(lists)
    np.save(path, frames)



cams_sn = {
    "cam0": "06955451",
    "cam1": "32052251",
    "cam2": "39050251",
    "cam3": "32050651"
}


fps = 200
gain = 5
shutter = shtr_spd(fps)

kR_cam0, kG_cam0, kB_cam0 = get_WB_coef(cams_sn["cam0"], 30, shutter, gain)
kR_cam1, kG_cam1, kB_cam1 = get_WB_coef(cams_sn["cam1"], 30, shutter, gain)
kR_cam2, kG_cam2, kB_cam2 = get_WB_coef(cams_sn["cam2"], 30, shutter, gain)
kR_cam3, kG_cam3, kB_cam3 = get_WB_coef(cams_sn["cam3"], 30, shutter, gain)

cam0, img0 = camera_init(cams_sn["cam0"], fps, shutter, gain)
cam1, img1 = camera_init(cams_sn["cam1"], fps, shutter, gain)
cam2, img2 = camera_init(cams_sn["cam2"], fps, shutter, gain)
cam3, img3 = camera_init(cams_sn["cam3"], fps, shutter, gain)

cam0.start_acquisition()
cam1.start_acquisition()
cam2.start_acquisition()
cam3.start_acquisition()


#TCP_IP = "169.254.226.95"
TCP_IP = "100.1.1.3"
TCP_PORT = 5005
buffer_size = 20

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((TCP_IP, TCP_PORT))
message_connect = "connected"
s.send(message_connect.encode())


while True:
    data_raw = s.recv(buffer_size)
    data = data_raw.decode()

    if "start" in data:
        name, status, timestamp = data.split("_")
        print(name, "start")

        metadata_cam0 = {
            "frame_timestamp": [],
            "framerate": fps,
            "shutter_speed": shutter,
            "gain": gain,
            "sn": cams_sn["cam0"],
            "WB_auto_coeff_RGB": [kR_cam0, kG_cam0, kB_cam0]
        }

        metadata_cam1 = {
            "frame_timestamp": [],
            "framerate": fps,
            "shutter_speed": shutter,
            "gain": gain,
            "sn": cams_sn["cam1"],
            "WB_auto_coeff_RGB": [kR_cam1, kG_cam1, kB_cam1]
        }

        metadata_cam2 = {
            "frame_timestamp": [],
            "framerate": fps,
            "shutter_speed": shutter,
            "gain": gain,
            "sn": cams_sn["cam2"],
            "WB_auto_coeff_RGB": [kR_cam2, kG_cam2, kB_cam2]
        }

        metadata_cam3 = {
            "frame_timestamp": [],
            "framerate": fps,
            "shutter_speed": shutter,
            "gain": gain,
            "sn": cams_sn["cam3"],
            "WB_auto_coeff_RGB": [kR_cam3, kG_cam3, kB_cam3]
        }

        cam0_l = []
        cam1_l = []
        cam2_l = []
        cam3_l = []

        counter = 0

        start = time.monotonic()
        s.setblocking(0)
        while True:
            try:
                data_raw = s.recv(buffer_size)
                data = data_raw.decode()
            except socket.error:
                pass
            counter += 1
            cam0.get_image(img0)
            co0 = img0.get_image_data_numpy()
            cam0_l.append(co0)
            metadata_cam0["frame_timestamp"].append(time.monotonic())

            cam1.get_image(img1)
            co1 = img1.get_image_data_numpy()
            cam1_l.append(co1)
            metadata_cam1["frame_timestamp"].append(time.monotonic())
            
            cam2.get_image(img2)
            co2 = img2.get_image_data_numpy()
            cam2_l.append(co2)
            metadata_cam2["frame_timestamp"].append(time.monotonic())

            cam3.get_image(img3)
            co3 = img3.get_image_data_numpy()
            cam3_l.append(co3)
            metadata_cam3["frame_timestamp"].append(time.monotonic())
            
            
            # print(counter)
            if "stop" in data:
                s.setblocking(1)
                break

        stop = time.monotonic()
        print("recorded_in", stop - start, counter)

        start_x = time.monotonic()
        total_rec = start_x - start
        for ix, v in enumerate([cam0_l, cam1_l, cam2_l, cam3_l]):
            filename = "{}_cam{}_frames-{}_trial-{}_{}".format(
                name,
                ix,
                str(len(v)).zfill(4),
                str(counter).zfill(3),
                timestamp
            )
            npy_path = op.join("data", filename + ".npy")
            json_path = op.join("data", filename + ".json")
            dump_and_run(v, npy_path)
            with open(json_path, "w") as fp:
                json.dump([metadata_cam0, metadata_cam1, metadata_cam2, metadata_cam3][ix], fp)
        stop_x = time.monotonic()
        dump_time = stop_x - start_x
        print("DATA DUMP IN:", dump_time)
        message_dump = "dumped_{}_rec_{}".format(dump_time, total_rec)
        s.send(message_dump.encode())
    if "exit" in data:
        break

s.close()
