import numpy as np
from ximea import xiapi
import cv2

framerate = 60.0
shutter = int((1 / framerate) * 1e+6) - 100
gain = 5
f_size = (1280, 1024)
img_format = "XI_RGB32"

cam_sns = ["06955451","32052251","39050251","32050651"]
cams = []
imgs = []
cam_lists = {}
for sn in cam_sns:
    ############################
    # for each camera separately
    cam = xiapi.Camera()
    cam.open_device_by_SN(sn)  # put a serial number of the camera
    cam.set_exposure(shutter)
    cam.set_gain(gain)
    cam.set_acq_timing_mode("XI_ACQ_TIMING_MODE_FRAME_RATE")
    cam.set_framerate(framerate)
    cam.set_imgdataformat(img_format)
    cam.enable_auto_wb()
    img = xiapi.Image()
    cam.start_acquisition()
    cams.append(cam)
    imgs.append(img)
    cam_lists[sn] = []

############################

while True:
    try:
        cam_datas = []
        for sn, cam, img in zip(cam_sns, cams, imgs):
            cam.get_image(img)

            cam_data = img.get_image_data_numpy()
            cam_lists[sn].append(cam_data[:, :, :3])

            width = int(f_size[0] * .5)
            height = int(f_size[1] * .5)
            dim = (width, height)
            resized = cv2.resize(cam_data, dim, interpolation=cv2.INTER_AREA)
            cam_datas.append(resized)

        data = np.vstack([np.hstack([cam_datas[0], cam_datas[1]]), np.hstack([cam_datas[2], cam_datas[2]])])

        cv2.imshow("cam", data)
        cv2.waitKey(1)

    except KeyboardInterrupt:
        for cam in cams:
            cam.stop_acquisition()
            cam.close_device()
        break
