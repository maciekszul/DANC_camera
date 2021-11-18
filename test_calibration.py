import matplotlib.pyplot as plt
import pickle
import numpy as np
from ximea import xiapi
import cv2

from utilities.calib_tools import locate, locate_dlt

framerate = 60.0
shutter = int((1 / framerate) * 1e+6) - 100
gain = 5
f_size = (1280, 1024)
img_format = "XI_RGB32"

# Rows and columns in chess board
cbcol = 9
cbrow = 6

cam_sns = ["06955451","32052251","39050251","32050651"]
cams = []
imgs = []

############################
# for each camera separately
for sn in cam_sns:
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

# Load intrinsic parameters
handle = open('intrinsic_params.pickle', "rb")
intrinsic_params = pickle.load(handle)
handle.close()

# Load extrinsic parameters
handle = open('extrinsic_params.pickle', "rb")
extrinsic_params = pickle.load(handle)
handle.close()

# Initialize plots
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection="3d")
xlim = [0, 1]
ylim = [0, 1]
zlim = [0, 1]

ax.set_xlim(xlim[0], xlim[1])
ax.set_ylim(ylim[0], ylim[1])
ax.set_zlim(zlim[0], zlim[1])

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.draw()
plt.pause(0.001)

points_per_image = cbcol * cbrow

while True:
    try:
        # Frame from each camera
        cam_datas = []
        # Chessboard corners from each camera
        cam_coords = {}

        # For each camera
        for sn, cam, img in zip(cam_sns, cams, imgs):

            # Get image
            cam.get_image(img)
            cam_data = img.get_image_data_numpy()

            # Find chessboard corners - fast
            gray = cv2.cvtColor(cam_data, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (cbcol, cbrow), None, flags=cv2.CALIB_CB_FAST_CHECK)
            cam_coords[sn] = []

            # If chesssboard detected
            if ret:
                cam_coords[sn] = corners

                # Visual feedback
                cam_data = cv2.drawChessboardCorners(cam_data, (cbcol, cbrow), corners, ret)
                cam_data = cv2.rectangle(cam_data, (5, 5), (cam_data.shape[1] - 5, cam_data.shape[0] - 5), (0, 255, 0),
                                         5)

            # Resize for display
            width = int(f_size[0] * .5)
            height = int(f_size[1] * .5)
            dim = (width, height)
            resized = cv2.resize(cam_data, dim, interpolation=cv2.INTER_AREA)
            cam_datas.append(resized)

        ax.clear()
        
        # Localize
        for idx in range(cbcol*cbrow):
            img_points={}
            for sn in cam_sns:
                if len(cam_coords[sn]):
                    img_points[sn]=cam_coords[sn][idx]
                else:
                    img_points[sn]=[]
            #[location, pairs_used] = locate(cam_sns, img_points, intrinsic_params, extrinsic_params)
            [location, pairs_used] = locate_dlt(cam_sns, img_points, intrinsic_params, extrinsic_params)

            # Plot 3d coordinates localized without SBA
            if pairs_used > 0:
                xs = location[:, 0]
                ys = location[:, 1]
                zs = location[:, 2]
                ax.scatter(xs, ys, zs, c='g', marker='o', s=1)
                xlim = [min(xlim[0], np.min(xs)), max(xlim[1], np.max(xs))]
                ylim = [min(ylim[0], np.min(ys)), max(ylim[1], np.max(ys))]
                zlim = [min(zlim[0], np.min(zs)), max(zlim[1], np.max(zs))]

        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_zlim(zlim[0], zlim[1])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.draw()
        plt.pause(0.001)

        data = np.vstack([np.hstack([cam_datas[0], cam_datas[1]]), np.hstack([cam_datas[2], cam_datas[2]])])
        cv2.imshow("cam", data)
        cv2.waitKey(1)

    except KeyboardInterrupt:
        for cam in cams:
            cam.stop_acquisition()
            cam.close_device()
        break
