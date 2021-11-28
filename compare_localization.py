import json
import sys

import matplotlib.pyplot as plt
import pickle
import numpy as np
from cv2 import aruco

from camera_io import shtr_spd, init_camera_sources
from utilities.tools import quick_resize
import cv2

from utilities.calib_tools import locate, locate_sba, locate_dlt, DoubleCharucoBoard

# Initialize ArUco Tracking
detect_parameters = aruco.DetectorParameters_create()
detect_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG

subcorner_term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

fps = 60.0
shutter = shtr_spd(fps)
gain = 5
f_size = (1280, 1024)

intrinsic_file = sys.argv[1]
extrinsic_file = sys.argv[2]
sba_file = sys.argv[3]

try:
    json_file = sys.argv[4]
    print("USING: ", json_file)
except:
    json_file = "settings.json"
    print("USING: ", json_file)

# opening a json file
with open(json_file) as settings_file:
    params = json.load(settings_file)

cams = init_camera_sources(params, fps, shutter, gain)

# Load intrinsic parameters
handle = open(intrinsic_file, "rb")
intrinsic_params = pickle.load(handle)
handle.close()

# Load extrinsic parameters
handle = open(extrinsic_file, "rb")
extrinsic_params = pickle.load(handle)
handle.close()

# Load SBA extrinsic parameters
handle = open(sba_file, "rb")
extrinsic_sba_params = pickle.load(handle)
handle.close()

board = DoubleCharucoBoard()
axis_size = 0.025  # This value is in meters

# Initialize plots
fig = plt.figure()
ax1 = fig.add_subplot(1, 3, 1, projection="3d")
ax2 = fig.add_subplot(1, 3, 2, projection="3d")
ax3 = fig.add_subplot(1, 3, 3, projection="3d")
xlim1 = [-0.001, 0.001]
ylim1 = [-0.001, 0.001]
zlim1 = [-0.001, 0.001]
xlim2 = [-0.001, 0.001]
ylim2 = [-0.001, 0.001]
zlim2 = [-0.001, 0.001]
xlim3 = [-0.001, 0.001]
ylim3 = [-0.001, 0.001]
zlim3 = [-0.001, 0.001]

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

while True:
    try:
        # Frame from each camera
        cam_datas = []
        # Chessboard corners from each camera
        cam_outside_corners = {}
        cam_inside_corners = {}

        # For each camera
        for cam in cams:

            # Get image
            cam_data = cam.next_frame()[:, :, :3].astype(np.uint8)

            k = intrinsic_params[cam.sn]['k']
            d = intrinsic_params[cam.sn]['d']

            gray = cv2.cvtColor(cam_data, cv2.COLOR_BGR2GRAY)

            cam_outside_corners[cam.sn] = np.array([])
            cam_inside_corners[cam.sn] = np.array([])

            [marker_corners, marker_ids, _] = cv2.aruco.detectMarkers(gray, board.dictionary,
                                                                      parameters=detect_parameters)
            cam_board = board.get_detected_board(marker_ids)

            if cam_board is not None and len(marker_corners) > 0:
                [ret, charuco_corners, charuco_ids] = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids,
                                                                                          gray, cam_board)
                if ret > 0:
                    charuco_corners_sub = cv2.cornerSubPix(gray, charuco_corners, (11, 11), (-1, -1),
                                                           subcorner_term_crit)

                    cam_data = cv2.rectangle(cam_data, (5, 5), (f_size[0] - 5, f_size[1] - 5), (0, 255, 0), 5)
                    cam_data = cv2.aruco.drawDetectedMarkers(cam_data.copy(), marker_corners, marker_ids)
                    cam_data = cv2.aruco.drawDetectedCornersCharuco(cam_data.copy(), charuco_corners_sub,
                                                                    charuco_ids)

                    # Estimate the posture of the charuco board, which is a construction of 3D space based on the 2D
                    # video
                    pose, rvec, tvec = aruco.estimatePoseCharucoBoard(
                        charucoCorners=charuco_corners_sub,
                        charucoIds=charuco_ids,
                        board=cam_board,
                        cameraMatrix=k,
                        distCoeffs=d,
                        rvec=None,
                        tvec=None
                    )
                    if pose:
                        cam_data = aruco.drawAxis(cam_data, k, d, rvec, tvec, axis_size)
                        cam_outside_corners[cam.sn], cam_inside_corners[cam.sn] = board.project(cam_board, k, d, rvec,
                                                                                                tvec)


            # Resize for display
            width = int(f_size[0] * .5)
            height = int(f_size[1] * .5)
            dim = (width, height)
            resized = cv2.resize(cam_data, dim, interpolation=cv2.INTER_AREA)
            cam_datas.append(resized)

        ax1.clear()
        ax2.clear()
        ax3.clear()

        # Mean triangulation
        outside_corner_locations = np.zeros((4, 3))
        for idx in range(4):
            img_points = {}
            for sn in cam_outside_corners.keys():
                if len(cam_outside_corners[sn]):
                    img_points[sn] = cam_outside_corners[sn][idx, :]
            [outside_corner_locations[idx, :], pairs_used] = locate(list(img_points.keys()), img_points,
                                                                    intrinsic_params,
                                                                    extrinsic_params)
        inside_corner_locations = np.zeros((board.n_square_corners, 3))
        for idx in range(board.n_square_corners):
            img_points = {}
            for sn in cam_inside_corners.keys():
                if len(cam_inside_corners[sn]):
                    img_points[sn] = cam_inside_corners[sn][idx, :]
            [inside_corner_locations[idx, :], pairs_used] = locate(list(img_points.keys()), img_points,
                                                                   intrinsic_params,
                                                                   extrinsic_params)

        board.plot_3d(ax1, outside_corner_locations, inside_corner_locations)
        xlim1 = [min(xlim1[0], np.min(outside_corner_locations[:, 0])),
                 max(xlim1[1], np.max(outside_corner_locations[:, 0]))]
        ylim1 = [min(ylim1[0], np.min(outside_corner_locations[:, 1])),
                 max(ylim1[1], np.max(outside_corner_locations[:, 1]))]
        zlim1 = [min(zlim1[0], np.min(outside_corner_locations[:, 2])),
                 max(zlim1[1], np.max(outside_corner_locations[:, 2]))]

        # SBA
        # outside_corner_locations = np.zeros((4, 3))
        # for idx in range(4):
        #     img_points = {}
        #     for sn in cam_outside_corners.keys():
        #         if len(cam_outside_corners[sn]):
        #             img_points[sn] = cam_outside_corners[sn][idx, :]
        #     [outside_corner_locations[idx, :], pairs_used] = locate_sba(list(img_points.keys()), img_points,
        #                                                                 intrinsic_params, extrinsic_params)
        # inside_corner_locations = np.zeros((board.n_square_corners, 3))
        # for idx in range(board.n_square_corners):
        #     img_points = {}
        #     for sn in cam_inside_corners.keys():
        #         if len(cam_inside_corners[sn]):
        #             img_points[sn] = cam_inside_corners[sn][idx, :]
        #     [inside_corner_locations[idx, :], pairs_used] = locate_sba(list(img_points.keys()), img_points,
        #                                                                intrinsic_params, extrinsic_params)
        #
        # board.plot_3d(ax2, outside_corner_locations, inside_corner_locations)
        # xlim2 = [min(xlim2[0], np.min(outside_corner_locations[:, 0])),
        #          max(xlim2[1], np.max(outside_corner_locations[:, 0]))]
        # ylim2 = [min(ylim2[0], np.min(outside_corner_locations[:, 1])),
        #          max(ylim2[1], np.max(outside_corner_locations[:, 1]))]
        # zlim2 = [min(zlim2[0], np.min(outside_corner_locations[:, 2])),
        #          max(zlim2[1], np.max(outside_corner_locations[:, 2]))]

        # DLT
        outside_corner_locations = np.zeros((4, 3))
        for idx in range(4):
            img_points = {}
            for sn in cam_outside_corners.keys():
                if len(cam_outside_corners[sn]):
                    img_points[sn] = cam_outside_corners[sn][idx, :]
            [outside_corner_locations[idx, :], pairs_used] = locate_dlt(list(img_points.keys()), img_points,
                                                                        intrinsic_params, extrinsic_params)
        inside_corner_locations = np.zeros((board.n_square_corners, 3))
        for idx in range(board.n_square_corners):
            img_points = {}
            for sn in cam_inside_corners.keys():
                if len(cam_inside_corners[sn]):
                    img_points[sn] = cam_inside_corners[sn][idx, :]
            [inside_corner_locations[idx, :], pairs_used] = locate_dlt(list(img_points.keys()), img_points,
                                                                       intrinsic_params, extrinsic_params)
        board.plot_3d(ax3, outside_corner_locations, inside_corner_locations)
        xlim3 = [min(xlim3[0], np.min(outside_corner_locations[:, 0])),
                 max(xlim3[1], np.max(outside_corner_locations[:, 0]))]
        ylim3 = [min(ylim3[0], np.min(outside_corner_locations[:, 1])),
                 max(ylim3[1], np.max(outside_corner_locations[:, 1]))]
        zlim3 = [min(zlim3[0], np.min(outside_corner_locations[:, 2])),
                 max(zlim3[1], np.max(outside_corner_locations[:, 2]))]

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

        # redraw the canvas
        fig.canvas.draw()
        # convert canvas to image
        plt_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plt_img = plt_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # img is rgb, convert to opencv's default bgr
        plt_img = cv2.cvtColor(plt_img, cv2.COLOR_RGB2BGR)

        ratio = resized.shape[0] / plt_img.shape[0]
        resized_plt = quick_resize(plt_img, ratio, plt_img.shape[1], plt_img.shape[0])

        if len(cam_datas) == 2:
            data = np.hstack([resized_plt, cam_datas[0], cam_datas[1]])
        elif len(cam_datas) == 3:
            blank_cam = np.ones(cam_datas[0].shape).astype(np.uint8)
            blank_plt = np.ones(resized_plt.shape).astype(np.uint8)
            data = np.vstack([np.hstack([resized_plt, cam_datas[0], cam_datas[1]]),
                              np.hstack([blank_plt, blank_cam, cam_datas[2]])])
        else:
            blank_plt = np.ones(resized_plt.shape).astype(np.uint8)
            data = np.vstack([np.hstack([resized_plt, cam_datas[0], cam_datas[1]]),
                              np.hstack([blank_plt, cam_datas[2], cam_datas[3]])])

        cv2.imshow("cam", data)
        cv2.waitKey(1)

    except KeyboardInterrupt:
        for cam in cams:
            cam.close()
        break
