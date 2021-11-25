import socket
import time
import json
import numpy as np
import os.path as op

from camera_io import init_camera_sources, shtr_spd, get_WB_coef


def dump_and_run(lists, path):
    frames = np.array(lists)
    np.save(path, frames)


# opening a json file
with open('settings.json') as settings_file:
    params = json.load(settings_file)

fps = 200
gain = 5
shutter = shtr_spd(fps)

kR_cam0, kG_cam0, kB_cam0 = get_WB_coef(params['cams_sn'][0], 30, shutter, gain)
kR_cam1, kG_cam1, kB_cam1 = get_WB_coef(params['cams_sn'][1], 30, shutter, gain)
kR_cam2, kG_cam2, kB_cam2 = get_WB_coef(params['cams_sn'][2], 30, shutter, gain)
kR_cam3, kG_cam3, kB_cam3 = get_WB_coef(params['cams_sn'][3], 30, shutter, gain)

cams=init_camera_sources(params, fps, shutter, gain, sensor_feature_value=1, disable_auto_bandwidth=True,
                         img_data_format='XI_RAW8', auto_wb=False, counter_selector='XI_CNT_SEL_API_SKIPPED_FRAMES')


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
            "sn": params['cam_sns'][0],
            "WB_auto_coeff_RGB": [kR_cam0, kG_cam0, kB_cam0]
        }

        metadata_cam1 = {
            "frame_timestamp": [],
            "framerate": fps,
            "shutter_speed": shutter,
            "gain": gain,
            "sn": params['cam_sns'][1],
            "WB_auto_coeff_RGB": [kR_cam1, kG_cam1, kB_cam1]
        }

        metadata_cam2 = {
            "frame_timestamp": [],
            "framerate": fps,
            "shutter_speed": shutter,
            "gain": gain,
            "sn": params['cam_sns'][2],
            "WB_auto_coeff_RGB": [kR_cam2, kG_cam2, kB_cam2]
        }

        metadata_cam3 = {
            "frame_timestamp": [],
            "framerate": fps,
            "shutter_speed": shutter,
            "gain": gain,
            "sn": params['cam_sns'][3],
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
