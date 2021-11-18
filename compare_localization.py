import matplotlib.pyplot as plt
import pickle
import numpy as np
from ximea import xiapi
import cv2

from utilities.calib_tools import locate, locate_sba, locate_dlt

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

# Load SBA extrinsic parameters
#handle = open('extrinsic_sba_params.pickle', "rb")
#extrinsic_sba_params = pickle.load(handle)
#handle.close()

# Initialize plots
fig = plt.figure()
ax1 = fig.add_subplot(1, 3, 1, projection="3d")
ax2 = fig.add_subplot(1, 3, 2, projection="3d")
ax3 = fig.add_subplot(1, 3, 3, projection="3d")
xlim1 = [0, 1]
ylim1 = [0, 1]
zlim1 = [0, 1]
xlim2 = [0, 1]
ylim2 = [0, 1]
zlim2 = [0, 1]
xlim3 = [0, 1]
ylim3 = [0, 1]
zlim3 = [0, 1]

ax1.set_xlim(xlim1[0], xlim1[1])
ax1.set_ylim(ylim1[0], ylim1[1])
ax1.set_zlim(zlim1[0], zlim1[1])
ax2.set_xlim(xlim2[0], xlim2[1])
ax2.set_ylim(ylim2[0], ylim2[1])
ax2.set_zlim(zlim2[0], zlim2[1])
ax3.set_xlim(xlim3[0], xlim3[1])
ax3.set_ylim(ylim3[0], ylim3[1])
ax3.set_zlim(zlim3[0], zlim3[1])

ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Z")
ax3.set_xlabel("X")
ax3.set_ylabel("Y")
ax3.set_zlabel("Z")
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

        ax1.clear()
        ax2.clear()
        ax3.clear()

        for idx in range(cbcol * cbrow):
            img_points = {}
            for sn in cam_sns:
                if len(cam_coords[sn]):
                    img_points[sn] = cam_coords[sn][idx]
                else:
                    img_points[sn]=[]
            # Localize
            [location1, pairs_used1] = locate(cam_sns, img_points, intrinsic_params, extrinsic_params)
            #[location2, pairs_used2] = locate_sba(cam_sns, img_points, intrinsic_params, extrinsic_sba_params)
            location2=location1
            pairs_used2=pairs_used1
            [location3, pairs_used3] = locate_dlt(cam_sns, img_points, intrinsic_params, extrinsic_params)


            # Plot 3d coordinates localized with average triangulation
            if pairs_used1 > 0:
                xs = location1[:, 0]
                ys = location1[:, 1]
                zs = location1[:, 2]
                ax1.scatter(xs, ys, zs, c='g', marker='o', s=1)
                xlim1 = [min(xlim1[0], np.min(xs)), max(xlim1[1], np.max(xs))]
                ylim1 = [min(ylim1[0], np.min(ys)), max(ylim1[1], np.max(ys))]
                zlim1 = [min(zlim1[0], np.min(zs)), max(zlim1[1], np.max(zs))]

            # Plot 3d coordinates localized with SBA
            if pairs_used2 > 0:
                xs = location2[:, 0]
                ys = location2[:, 1]
                zs = location2[:, 2]
                ax2.scatter(xs, ys, zs, c='g', marker='o', s=1)
                xlim2 = [min(xlim2[0], np.min(xs)), max(xlim2[1], np.max(xs))]
                ylim2 = [min(ylim2[0], np.min(ys)), max(ylim2[1], np.max(ys))]
                zlim2 = [min(zlim2[0], np.min(zs)), max(zlim2[1], np.max(zs))]

            # Plot 3d coordinates localized with DLT
            if pairs_used3 > 0:
                xs = location3[:, 0]
                ys = location3[:, 1]
                zs = location3[:, 2]
                ax3.scatter(xs, ys, zs, c='g', marker='o', s=1)
                xlim3 = [min(xlim3[0], np.min(xs)), max(xlim3[1], np.max(xs))]
                ylim3 = [min(ylim3[0], np.min(ys)), max(ylim3[1], np.max(ys))]
                zlim3 = [min(zlim3[0], np.min(zs)), max(zlim3[1], np.max(zs))]

        ax1.set_xlim(xlim1[0], xlim1[1])
        ax1.set_ylim(ylim1[0], ylim1[1])
        ax1.set_zlim(zlim1[0], zlim1[1])
        ax2.set_xlim(xlim2[0], xlim2[1])
        ax2.set_ylim(ylim2[0], ylim2[1])
        ax2.set_zlim(zlim2[0], zlim2[1])
        ax3.set_xlim(xlim3[0], xlim3[1])
        ax3.set_ylim(ylim3[0], ylim3[1])
        ax3.set_zlim(zlim3[0], zlim3[1])
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")
        ax3.set_xlabel("X")
        ax3.set_ylabel("Y")
        ax3.set_zlabel("Z")
        plt.draw()
        plt.pause(0.001)

        data = np.vstack([np.hstack([cam_datas[0], cam_datas[1]]), np.hstack([cam_datas[2], cam_datas[3]])])
        cv2.imshow("cam", data)
        cv2.waitKey(1)

    except KeyboardInterrupt:
        for cam in cams:
            cam.stop_acquisition()
            cam.close_device()
        break
