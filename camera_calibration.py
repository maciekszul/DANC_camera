import json
import sys

import numpy as np
from cv2 import aruco

import cv2
import pickle
import matplotlib.pyplot as plt

from camera_io import init_camera_sources, init_file_sources, shtr_spd
from utilities.calib_tools import locate, DoubleCharucoBoard
from utilities.tools import quick_resize

subcorner_term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
stereo_term_crit = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 1e-5)
intrinsic_flags = cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_PRINCIPAL_POINT
intrinsic_term_crit = (cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9)

# Initialize ArUco Tracking
detect_parameters = aruco.DetectorParameters_create()
detect_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG

fps = 60.0
shutter = shtr_spd(fps)
gain = 5
f_size = (1280, 1024)


def collect_sba_data(parameters, cams, intrinsic_params, extrinsic_params):
    """
    Collect data for sparse bundle adjustment
    :param parameters: Acquisition parameters
    :param cams: list of camera objects
    :param intrinsic_params: intrinsic calibration parameters for each camera
    :param extrinsic_params: extrinsic calibration parameters for each camera
    """

    if parameters['type'] == 'offline':
        cams = init_file_sources(parameters, 'sba')

    # Initialize array
    cam_list = {}
    for cam in cams:
        cam_list[cam.sn] = []
    points_3d = []
    point_3d_indices = []
    points_2d = []
    camera_indices = []

    board = DoubleCharucoBoard()
    axis_size = 0.025  # This value is in meters

    print('Accept SBA data (y/n)?')

    # Get corresponding points between images
    point_idx_counter = 0

    # Acquire enough good frames
    while True:
        # Frames for each camera
        cam_datas = []
        vcam_datas = []
        # Chessboard corners for each camera
        cam_corners = {}
        cam_corner_ids = {}

        # Go through each camera
        video_finished = True
        for cam_idx, cam in enumerate(cams):

            # Get image from camera
            cam_data = cam.next_frame()
            if cam_data is not None:
                video_finished = False
            else:
                break
            cam_datas.append(cam_data)
            vcam_data = np.copy(cam_data)[:, :, :3].astype(np.uint8)

            k = intrinsic_params[cam.sn]['k']
            d = intrinsic_params[cam.sn]['d']

            # Convert to greyscale for chess board detection
            gray = cv2.cvtColor(cam_data, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners - fast checking
            [marker_corners, marker_ids, _] = cv2.aruco.detectMarkers(gray, board.dictionary,
                                                                      parameters=detect_parameters)
            if len(marker_corners):
                cam_board = board.get_detected_board(marker_ids)

                [ret, charuco_corners, charuco_ids] = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids,
                                                                                          gray, cam_board)
                if ret > 0:
                    charuco_corners_sub = cv2.cornerSubPix(gray, charuco_corners, (11, 11), (-1, -1),
                                                           subcorner_term_crit)

                    vcam_data = cv2.rectangle(vcam_data, (5, 5), (f_size[0] - 5, f_size[1] - 5), (0, 255, 0), 5)
                    vcam_data = cv2.aruco.drawDetectedMarkers(vcam_data.copy(), marker_corners, marker_ids)
                    vcam_data = cv2.aruco.drawDetectedCornersCharuco(vcam_data.copy(), charuco_corners_sub,
                                                                     charuco_ids)

                    if ret > 20:

                        # Estimate the posture of the charuco board, which is a construction of 3D space based on the
                        # 2D video
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
                            vcam_data = aruco.drawAxis(vcam_data, k, d, rvec, tvec, axis_size)

                            pts = []
                            ids = []
                            for corner_id in range(board.n_square_corners):
                                c_id = board.get_corresponding_corner_id(corner_id, cam_board)

                                if len(np.where(charuco_ids == c_id)[0]):
                                    c_idx = np.where(charuco_ids == c_id)[0][0]
                                    pts.append(charuco_corners_sub[c_idx, :, :])
                                    ids.append(c_id)
                            cam_corners[cam.sn] = pts
                            cam_corner_ids[cam.sn] = np.array(ids)

            # Num frames
            vcam_data = cv2.putText(vcam_data, '%d frames' % len(cam_list[cam.sn]), (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                                    2, (0, 255, 0), 2)

            # Resize for display
            resized = quick_resize(vcam_data, 0.5, f_size[0], f_size[1])
            vcam_datas.append(resized)

            cam_list[cam.sn].append(cam_datas[cam_idx][:, :, :3])

        # If chessboard visible in more than one camera
        for board_id in range(board.n_square_corners):
            visible_cams = []
            for cam in cams:
                if cam.sn in cam_corner_ids and len(np.where(cam_corner_ids[cam.sn] == board_id)[0]):
                    visible_cams.append(cam.sn)

            if len(visible_cams) > 1:
                # Add 3d and 3d points to list
                c = {}
                for cam_idx, cam in enumerate(cams):
                    if cam.sn in visible_cams:
                        c_idx = np.where(cam_corner_ids[cam.sn] == board_id)[0][0]
                        points_2d.append(cam_corners[cam.sn][c_idx])
                        point_3d_indices.append(point_idx_counter)
                        camera_indices.append(cam_idx)
                        c[cam.sn] = cam_corners[cam.sn][c_idx]

                point_3d_est, paires_used = locate(visible_cams, c, intrinsic_params, extrinsic_params)

                points_3d.append(point_3d_est)
                point_idx_counter += 1

        if video_finished:
            break

        # Show camera images
        if len(vcam_datas) == 1:
            data = vcam_datas[0]
        elif len(vcam_datas) == 2:
            data = np.hstack([vcam_datas[0], vcam_datas[1]])
        elif len(vcam_datas) == 3:
            data = np.vstack([np.hstack([vcam_datas[0], vcam_datas[1]]), np.hstack([vcam_datas[2], vcam_datas[2]])])
        else:
            data = np.vstack([np.hstack([vcam_datas[0], vcam_datas[1]]), np.hstack([vcam_datas[2], vcam_datas[3]])])
        cv2.imshow("cam", data)
        key = cv2.waitKey(1) & 0xFF

        # Quit if enough frames
        if key == ord('y'):
            break
        # Start over
        elif key == ord('n'):
            # Initialize array
            cam_list = {}
            for cam in cams:
                cam_list[cam.sn] = []
            points_3d = []
            point_3d_indices = []
            points_2d = []
            camera_indices = []
            point_idx_counter = 0

    # Convert to numpy arrays
    points_2d = np.squeeze(np.array(points_2d, dtype=np.float32))
    points_3d = np.squeeze(np.array(points_3d, dtype=np.float32))
    point_3d_indices = np.array(point_3d_indices, dtype=np.int)
    camera_indices = np.array(camera_indices, dtype=np.int)

    # Save data for offline sparse bundle adjustment
    pickle.dump(
        {
            'points_2d': points_2d,
            'points_3d': points_3d,
            'points_3d_indices': point_3d_indices,
            'camera_indices': camera_indices
        },
        open(
            "sba_data.pickle",
            "wb",
        ),
    )

    # Save videos with frames used for sparse bundle adjustment
    for cam in cams:
        vid_list = cam_list[cam.sn]
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        cam_vid = cv2.VideoWriter(
            "sba_cam_{}.avi".format(cam.sn),
            fourcc,
            float(fps),
            f_size
        )
        [cam_vid.write(i) for i in vid_list]
        cam_vid.release()


def run_extrinsic_calibration(parameters, cams, intrinsic_params):
    """
    Run extrinsic calibration for each pair of cameras
    :param parameters: Acquisition parameters
    :param cams: list of camera objects
    :param intrinsic_params: intrinsic calibration parameters for each camera
    :return: extrinsic calibration parameters for each camera
    """

    # Rotation matrix for first camera - all others are relative to it
    extrinsic_params = {
        parameters['cam_sns'][0]: {
            'r': np.array([[1, 0, 0],
                           [0, 0, -1],
                           [0, 1, 0]], dtype=np.float32),
            't': np.array([[0, 0, 0]], dtype=np.float32).T
        }
    }

    # Go through each camera
    for cam1_idx in range(len(parameters['cam_sns'])):
        cam1_sn = parameters['cam_sns'][cam1_idx]

        # For every other camera (except pairs already calibrated)
        for cam2_idx in range(cam1_idx + 1, len(parameters['cam_sns'])):
            cam2_sn = parameters['cam_sns'][cam2_idx]

            if parameters['type'] == 'offline':
                cams = init_file_sources(parameters, 'extrinsic_%s-%s' % (cam1_sn, cam2_sn))
                cam1 = cams[0]
                cam2 = cams[1]
            else:
                cam1 = cams[cam1_idx]
                cam2 = cams[cam2_idx]

            # Calibrate pair
            print("Computing stereo calibration for cam %d and %d" % (cam1_idx, cam2_idx))
            rms, r, t, cam_list = extrinsic_cam_calibration(parameters, cam1, cam2, intrinsic_params, extrinsic_params)
            print(f"{rms:.5f} pixels")

            # https://en.wikipedia.org/wiki/Camera_resectioning#Extrinsic_parameters
            # T is the world origin position in the camera coordinates.
            # The world position of the camera is C = -(R^-1)@T.
            # Similarly, the rotation of the camera in world coordinates is given by R^-1
            extrinsic_params[cam2_sn] = {
                'r': r @ extrinsic_params[cam1_sn]['r'],
                't': r @ extrinsic_params[cam1_sn]['t'] + t
            }

            # Save frames used to calibrate for cam1
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            cam1_vid = cv2.VideoWriter(
                "extrinsic_{}-{}_cam_{}.avi".format(cam1_sn, cam2_sn, cam1_sn),
                fourcc,
                float(fps),
                f_size
            )
            [cam1_vid.write(i) for i in cam_list[cam1_sn]]
            cam1_vid.release()

            # Save frames used to calibrate for cam2
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            cam2_vid = cv2.VideoWriter(
                "extrinsic_{}-{}_cam_{}.avi".format(cam1_sn, cam2_sn, cam2_sn),
                fourcc,
                float(fps),
                f_size
            )
            [cam2_vid.write(i) for i in cam_list[cam2_sn]]
            cam2_vid.release()

    # Save extrinsic calibration parameters
    pickle.dump(
        extrinsic_params,
        open(
            "extrinsic_params.pickle",
            "wb",
        ),
    )
    return extrinsic_params


def extrinsic_cam_calibration(parameters, cam1, cam2, intrinsic_params, extrinsic_params):
    """
    Run extrinsic calibration for one pair of cameras
    :param parameters: Acquisition parameters
    :param cam1: camera 1 object
    :param cam2: camera 2 object
    :param intrinsic_params: intrinsic calibration parameters for each camera
    :param extrinsic_params: extrinsic calibration parameters for each camera
    :return: tuple: RMS, rotation matrix, translation matrix, list of frames for each camera
    """

    # Initialize
    cam_list = {cam1.sn: [], cam2.sn: []}
    objpoints = []
    imgpoints = {cam1.sn: [], cam2.sn: []}
    r = np.array([[1, 0, 0],
                  [0, 0, -1],
                  [0, 1, 0]], dtype=np.float32)
    t = np.array([[0, 0, 0]], dtype=np.float32).T

    axis_size = 0.025  # This value is in meters

    board = DoubleCharucoBoard()

    # Stop when RMS lower than threshold and greater than min frames collected
    rms_threshold = 1.0
    min_frames = 50
    rms = 1e6
    rmss = []

    # Get intrinsic parameters for each camera
    k1 = intrinsic_params[cam1.sn]['k']
    d1 = intrinsic_params[cam1.sn]['d']
    k2 = intrinsic_params[cam2.sn]['k']
    d2 = intrinsic_params[cam2.sn]['d']

    if parameters['type'] == 'online':
        print('Accept extrinsic calibration (y/n)?')

    # Plot 3D triangulation and RMSE
    fig = plt.figure('extrinsic - %s - %s' % (cam1.sn, cam2.sn))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)

    # Initialize axes limits
    xlim = [-0.001, 0.001]
    ylim = [-0.001, 0.001]
    zlim = [-0.001, 0.001]
    ax1.set_xlim(xlim[0], xlim[1])
    ax1.set_ylim(ylim[0], ylim[1])
    ax1.set_zlim(zlim[0], zlim[1])

    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax2.set_xlabel("frame")
    ax2.set_ylabel("RMS")

    # Acquire enough good frames - until RMSE low enough at at least min frames collected
    while rms > rms_threshold or len(imgpoints[cam1.sn]) < min_frames:
        # Get image from cam1 and 2
        cam1_data = cam1.next_frame()
        cam2_data = cam2.next_frame()

        if cam1_data is None or cam2_data is None:
            break

        vcam1_data = np.copy(cam1_data)[:, :, :3].astype(np.uint8)
        vcam2_data = np.copy(cam2_data)[:, :, :3].astype(np.uint8)

        # Convert to greyscale for chess board detection
        gray1 = cv2.cvtColor(cam1_data, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(cam2_data, cv2.COLOR_BGR2GRAY)
        img_shape1 = gray1.shape[::-1]

        # Find the chess board corners
        [marker_corners1, marker_ids1, _] = cv2.aruco.detectMarkers(gray1, board.dictionary,
                                                                    parameters=detect_parameters)
        cam1_board = board.get_detected_board(marker_ids1)

        [marker_corners2, marker_ids2, _] = cv2.aruco.detectMarkers(gray2, board.dictionary,
                                                                    parameters=detect_parameters)
        cam2_board = board.get_detected_board(marker_ids2)

        # If found, add object points, image points (after refining them)
        ret1 = 0
        if len(marker_corners1) and cam1_board is not None:
            [ret1, charuco_corners1, charuco_ids1] = cv2.aruco.interpolateCornersCharuco(marker_corners1, marker_ids1,
                                                                                         gray1, cam1_board)
            if ret1 > 0:
                charuco_corners_sub1 = cv2.cornerSubPix(gray1, charuco_corners1, (11, 11), (-1, -1),
                                                        subcorner_term_crit)

                vcam1_data = cv2.rectangle(vcam1_data, (5, 5), (f_size[0] - 5, f_size[1] - 5), (0, 255, 0), 5)
                vcam1_data = cv2.aruco.drawDetectedMarkers(vcam1_data.copy(), marker_corners1, marker_ids1)
                vcam1_data = cv2.aruco.drawDetectedCornersCharuco(vcam1_data.copy(), charuco_corners_sub1, charuco_ids1)

        ret2 = 0
        if len(marker_corners2) > 0 and cam2_board is not None:
            [ret2, charuco_corners2, charuco_ids2] = cv2.aruco.interpolateCornersCharuco(marker_corners2, marker_ids2,
                                                                                         gray2, cam2_board)

            if ret2 > 0:
                charuco_corners_sub2 = cv2.cornerSubPix(gray2, charuco_corners2, (11, 11), (-1, -1),
                                                        subcorner_term_crit)

                vcam2_data = cv2.rectangle(vcam2_data, (5, 5), (f_size[0] - 5, f_size[1] - 5), (0, 255, 0), 5)
                vcam2_data = cv2.aruco.drawDetectedMarkers(vcam2_data.copy(), marker_corners2, marker_ids2)
                vcam2_data = cv2.aruco.drawDetectedCornersCharuco(vcam2_data.copy(), charuco_corners_sub2, charuco_ids2)

        if ret1 > 20 and ret2 > 20:

            # Estimate the posture of the charuco board, which is a construction of 3D space based on the 2D video
            pose1, rvec1, tvec1 = aruco.estimatePoseCharucoBoard(
                charucoCorners=charuco_corners_sub1,
                charucoIds=charuco_ids1,
                board=cam1_board,
                cameraMatrix=k1,
                distCoeffs=d1,
                rvec=None,
                tvec=None
            )
            if pose1:
                vcam1_data = aruco.drawAxis(vcam1_data, k1, d1, rvec1, tvec1, axis_size)

            pose2, rvec2, tvec2 = aruco.estimatePoseCharucoBoard(
                charucoCorners=charuco_corners_sub2,
                charucoIds=charuco_ids2,
                board=cam2_board,
                cameraMatrix=k2,
                distCoeffs=d2,
                rvec=None,
                tvec=None
            )
            if pose2:
                vcam2_data = aruco.drawAxis(vcam2_data, k2, d2, rvec2, tvec2, axis_size)

            if pose1 and pose2:

                corners = []
                pts1 = []
                pts2 = []
                n_pts = 0
                for cam_corner_id in range(board.n_square_corners):
                    cam1_corner_id = board.get_corresponding_corner_id(cam_corner_id, cam1_board)
                    cam2_corner_id = board.get_corresponding_corner_id(cam_corner_id, cam2_board)

                    if len(np.where(charuco_ids1 == cam1_corner_id)[0]) and \
                            len(np.where(charuco_ids2 == cam2_corner_id)[0]):
                        corners.append(cam1_board.chessboardCorners[cam1_corner_id, :])

                        c1_idx=np.where(charuco_ids1==cam1_corner_id)[0][0]
                        pts1.append(charuco_corners_sub1[c1_idx, :, :])

                        c2_idx = np.where(charuco_ids2 == cam2_corner_id)[0][0]
                        pts2.append(charuco_corners_sub2[c2_idx, :, :])
                        n_pts = n_pts + 1

                if len(corners):
                    # Add to list of object points
                    objpoints.append(np.array(corners).reshape((n_pts, 3)).astype(np.float32))
                    imgpoints[cam1.sn].append(np.array(pts1).astype(np.float32))
                    imgpoints[cam2.sn].append(np.array(pts2).astype(np.float32))

                    # Stereo calibration - keep intrinsic parameters fixed
                    rms, *_, r_new, t_new, _, _ = cv2.stereoCalibrate(objpoints, imgpoints[cam1.sn], imgpoints[cam2.sn],
                                                                      k1, d1, k2, d2, img_shape1,
                                                                      flags=cv2.CALIB_FIX_INTRINSIC,
                                                                      criteria=stereo_term_crit)
                    # Mean RMSE
                    n_pts = []
                    for obj in objpoints:
                        n_pts.append(obj.shape[0])
                    rms = rms / np.mean(n_pts)

                    # If there is a jump in RMSE - exclude this point
                    if len(rmss) > 0 and rms - rmss[-1] > 5:
                        objpoints.pop()
                        imgpoints[cam1.sn].pop()
                        imgpoints[cam2.sn].pop()

                    # Otherwise, update lists and extrinsic parameters
                    else:
                        rmss.append(rms)
                        r = r_new
                        t = t_new

                        extrinsic_params[cam2.sn] = {
                            'r': r @ extrinsic_params[cam1.sn]['r'],
                            't': r @ extrinsic_params[cam1.sn]['t'] + t
                        }

                        cam_list[cam1.sn].append(cam1_data[:, :, :3])
                        cam_list[cam2.sn].append(cam2_data[:, :, :3])

                    # Triangulate
                    ax1.clear()
                    cam_outside_corners = {}
                    cam_inside_corners = {}
                    cam_outside_corners[cam1.sn], cam_inside_corners[cam1.sn] = board.project(cam1_board, k1, d1, rvec1, tvec1)
                    cam_outside_corners[cam2.sn], cam_inside_corners[cam2.sn] = board.project(cam2_board, k2, d2, rvec2, tvec2)

                    outside_corner_locations = np.zeros((4, 3))
                    for idx in range(4):
                        img_points = {}
                        for sn in cam_outside_corners.keys():
                            if len(cam_outside_corners[sn]):
                                img_points[sn] = cam_outside_corners[sn][idx, :]
                        [outside_corner_locations[idx, :], pairs_used] = locate(list(img_points.keys()), img_points,
                                                                                intrinsic_params,
                                                                                extrinsic_params)
                    inside_corner_locations = np.zeros((63, 3))
                    for idx in range(board.n_square_corners):
                        img_points = {}
                        for sn in cam_inside_corners.keys():
                            if len(cam_inside_corners[sn]):
                                img_points[sn] = cam_inside_corners[sn][idx, :]
                        [inside_corner_locations[idx, :], pairs_used] = locate(list(img_points.keys()), img_points,
                                                                               intrinsic_params,
                                                                               extrinsic_params)

                    board.plot_3d(ax1, outside_corner_locations, inside_corner_locations)
                    xlim = [min(xlim[0], np.min(outside_corner_locations[:, 0])),
                            max(xlim[1], np.max(outside_corner_locations[:, 0]))]
                    ylim = [min(ylim[0], np.min(outside_corner_locations[:, 1])),
                            max(ylim[1], np.max(outside_corner_locations[:, 1]))]
                    zlim = [min(zlim[0], np.min(outside_corner_locations[:, 2])),
                            max(zlim[1], np.max(outside_corner_locations[:, 2]))]
                    ax1.set_xlim(xlim[0], xlim[1])
                    ax1.set_ylim(ylim[0], ylim[1])
                    ax1.set_zlim(zlim[0], zlim[1])
                    ax1.set_xlabel("X")
                    ax1.set_ylabel("Y")
                    ax1.set_zlabel("Z")

        # Plot RMSE
        ax2.clear()
        ax2.plot(range(len(imgpoints[cam1.sn])), rmss)
        ax2.set_xlabel("frame")
        ax2.set_ylabel("RMS")

        # redraw the canvas
        fig.canvas.draw()
        # convert canvas to image
        plt_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plt_img = plt_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # img is rgb, convert to opencv's default bgr
        plt_img = cv2.cvtColor(plt_img, cv2.COLOR_RGB2BGR)

        # Show num frames so far
        vcam1_data = cv2.putText(vcam1_data, '%d frames' % len(imgpoints[cam1.sn]), (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                                 2,
                                 (0, 255, 0), 2)
        vcam2_data = cv2.putText(vcam2_data, '%d frames' % len(imgpoints[cam2.sn]), (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                                 2,
                                 (0, 255, 0), 2)

        # Resize for display
        resized1 = quick_resize(vcam1_data, 0.5, f_size[0], f_size[1])
        resized2 = quick_resize(vcam2_data, 0.5, f_size[0], f_size[1])
        ratio = resized1.shape[0]/plt_img.shape[0]
        resized_plt = quick_resize(plt_img, ratio, plt_img.shape[1], plt_img.shape[0])

        # Show camera images side by side
        data = np.hstack([resized_plt, resized1, resized2])
        cv2.imshow("cam", data)
        key = cv2.waitKey(1) & 0xFF

        # Quit if enough frames
        if key == ord('y'):
            break
        # Start over
        elif key == ord('n'):
            cam_list = {cam1.sn: [], cam2.sn: []}
            objpoints = []
            imgpoints = {cam1.sn: [], cam2.sn: []}
            rmss = []
            xlim = [-0.001, 0.001]
            ylim = [-0.001, 0.001]
            zlim = [-0.001, 0.001]

    if len(objpoints) > 0:
        # Final stereo calibration - keep intrinsic parameters fixed
        (rms, *_, r, t, _, _) = cv2.stereoCalibrate(objpoints, imgpoints[cam1.sn], imgpoints[cam2.sn],
                                                    k1, d1, k2, d2, img_shape1, flags=cv2.CALIB_FIX_INTRINSIC,
                                                    criteria=stereo_term_crit)
        # Mean RMSE
        n_pts = []
        for obj in objpoints:
            n_pts.append(obj.shape[0])
        rms = rms / np.mean(n_pts)

    return rms, r, t, cam_list


def run_intrinsic_calibration(parameters, cams):
    """
    Run intrinsic calibration for each camera
    :param parameters: Acquisition parameters
    :param cams: list of camera objects
    :return: intrinsic calibration parameters for each camera
    """
    intrinsic_params = {}

    if parameters['type'] == 'offline':
        cams = init_file_sources(parameters, 'intrinsic')

    # Calibrate each camera
    for cam in cams:
        print('Calibrating camera %s' % cam.sn)
        rpe, k, d, cam_list = intrinsic_cam_calibration(cam)
        print("Mean re-projection error: %.3f pixels " % rpe)
        intrinsic_params[cam.sn] = {
            'k': k,
            'd': d
        }

        # Save frames for calibration
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        cam_vid = cv2.VideoWriter(
            "intrinsic_cam_{}.avi".format(cam.sn),
            fourcc,
            float(fps),
            f_size
        )
        [cam_vid.write(i) for i in cam_list]
        cam_vid.release()

    # Save intrinsic parameters
    pickle.dump(
        intrinsic_params,
        open(
            "intrinsic_params.pickle",
            "wb",
        ),
    )
    return intrinsic_params


def intrinsic_cam_calibration(cam):
    """
    Run intrinsic calibration for one camera
    :param cam: camera object
    :return: tuple: reprojection error, k, distortion coefficient, list of frames for each camera
    """

    # Stop when RPE lower than threshold and greater than min frames collected
    rpe_threshold = 0.5
    min_frames = 50
    rpe = 1e6

    board = DoubleCharucoBoard()

    # Plot RPE
    fig = plt.figure('Intrinsic - %s' % cam.sn)
    ax = fig.add_subplot(111)
    ax.set_xlabel("frame")
    ax.set_ylabel("RPE")

    cam_list = []
    all_corners = []
    all_ids = []
    rpes = []

    # Initialize params
    k = np.eye(3)
    d = np.zeros((5, 1))

    map_x, map_y = cv2.initUndistortRectifyMap(k, d, None, k, (f_size[0], f_size[1]), cv2.CV_32FC1)

    # Acquire enough good frames - until RPE low enough at at least min frames collected
    while rpe > rpe_threshold or len(all_corners) < min_frames:

        # Get image from camera
        new_cam_data = cam.next_frame()
        if new_cam_data is None:
            break
        cam_data = new_cam_data
        vcam_data = np.copy(cam_data)[:, :, :3].astype(np.uint8)

        # Convert to greyscale for chess board detection
        gray = cv2.cvtColor(cam_data, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        [marker_corners, marker_ids, _] = cv2.aruco.detectMarkers(gray, board.dictionary, parameters=detect_parameters)

        cam_board = board.get_detected_board(marker_ids)

        # If found, add object points, image points (after refining them)
        if len(marker_corners) > 0 and cam_board is not None:
            img_shape = gray.shape[::-1]

            [ret, charuco_corners, charuco_ids] = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, gray,
                                                                                      cam_board)

            if ret > 20:
                charuco_corners_sub = cv2.cornerSubPix(gray, charuco_corners, (11, 11), (-1, -1), subcorner_term_crit)

                all_corners.append(charuco_corners_sub)
                all_ids.append(charuco_ids)

                # Visual feedback
                vcam_data = cv2.rectangle(vcam_data, (5, 5), (f_size[0] - 5, f_size[1] - 5), (0, 255, 0), 5)
                vcam_data = cv2.aruco.drawDetectedMarkers(vcam_data.copy(), marker_corners, marker_ids)
                vcam_data = cv2.aruco.drawDetectedCornersCharuco(vcam_data.copy(), charuco_corners_sub, charuco_ids)

                # Intrinsic calibration
                if len(all_corners) >= 6:
                    rpe, k, d, r, t = cv2.aruco.calibrateCameraCharuco(charucoCorners=all_corners,
                                                                       charucoIds=all_ids,
                                                                       board=cam_board,
                                                                       imageSize=img_shape,
                                                                       cameraMatrix=None,
                                                                       distCoeffs=None,
                                                                       flags=intrinsic_flags,
                                                                       criteria=intrinsic_term_crit)
                    # If there is a jump in RPE - exclude this point
                    if len(rpes) > 0 and rpe - rpes[-1] > 1:
                        all_corners.pop()
                        all_ids.pop()
                    # Otherwise, update lists and undistort rectify map
                    else:
                        rpes.append(rpe)
                        cam_list.append(cam_data[:, :, :3])
                        map_x, map_y = cv2.initUndistortRectifyMap(k, d, None, k, img_shape, cv2.CV_32FC1)

        undist_data = cv2.remap(vcam_data, map_x, map_y, cv2.INTER_LINEAR)
        undist_data = cv2.putText(undist_data, 'undistorted', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

        vcam_data = cv2.putText(vcam_data, '%d frames' % len(all_corners), (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                (0, 255, 0), 2)

        # Plot projection error
        ax.clear()
        ax.plot(range(len(rpes)), rpes)
        ax.set_xlabel("frame")
        ax.set_ylabel("RPE")

        # redraw the canvas
        fig.canvas.draw()
        # convert canvas to image
        plt_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plt_img = plt_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # img is rgb, convert to opencv's default bgr
        plt_img = cv2.cvtColor(plt_img, cv2.COLOR_RGB2BGR)

        # Resize for display
        resized = quick_resize(vcam_data, 0.5, f_size[0], f_size[1])
        undist_resized = quick_resize(undist_data, 0.5, f_size[0], f_size[1])
        ratio = resized.shape[0] / plt_img.shape[0]
        resized_plt = quick_resize(plt_img, ratio, plt_img.shape[1], plt_img.shape[0])

        data = np.hstack([resized_plt, resized, undist_resized])

        # Show camera image and undistorted image side by side
        cv2.imshow("cam", data)
        key = cv2.waitKey(1) & 0xFF

        # Quit if enough frames
        if key == ord('y'):
            break
        # Start over
        elif key == ord('n'):
            cam_list = []
            all_corners = []
            all_ids = []
            rpes = []

    # Final camera calibration
    rpe, k, d, r, t = cv2.aruco.calibrateCameraCharuco(charucoCorners=all_corners,
                                                       charucoIds=all_ids,
                                                       board=board,
                                                       imageSize=img_shape,
                                                       cameraMatrix=None,
                                                       distCoeffs=None,
                                                       flags=intrinsic_flags,
                                                       criteria=intrinsic_term_crit)

    return rpe, k, d, cam_list


def verify_calibration(cams, intrinsic_params, extrinsic_params):
    """
    Verify calibration
    :param cams: list of camera objects
    :param intrinsic_params: intrinsic calibration parameters for each camera
    :param extrinsic_params: extrinsic calibration parameters for each camera
    :return: whether or not to accept calibration
    """

    axis_size = 0.025  # This value is in meters

    board = DoubleCharucoBoard()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    xlim = [-0.001, 0.001]
    ylim = [-0.001, 0.001]
    zlim = [-0.001, 0.001]
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_zlim(zlim[0], zlim[1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    print('Accept final calibration (y/n)?')

    while True:
        cam_datas = []
        cam_outside_corners = {}
        cam_inside_corners = {}

        for cam in cams:
            cam_data = cam.next_frame()[:, :, :3].astype(np.uint8)
            k = intrinsic_params[cam.sn]['k']
            d = intrinsic_params[cam.sn]['d']

            gray = cv2.cvtColor(cam_data, cv2.COLOR_BGR2GRAY)

            cam_outside_corners[cam.sn] = np.array([])
            cam_inside_corners[cam.sn] = np.array([])

            [marker_corners, marker_ids, _] = cv2.aruco.detectMarkers(gray, board.dictionary,
                                                                      parameters=detect_parameters)
            cam_board = board.get_detected_board(marker_ids)

            if len(marker_corners)>0 and cam_board is not None:

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

            resized = quick_resize(cam_data, 0.5, f_size[0], f_size[1])
            cam_datas.append(resized)

        ax.clear()
        outside_corner_locations = np.zeros((4, 3))
        for idx in range(4):
            img_points = {}
            for sn in cam_outside_corners.keys():
                if len(cam_outside_corners[sn]):
                    img_points[sn] = cam_outside_corners[sn][idx, :]
            [outside_corner_locations[idx, :], _] = locate(list(img_points.keys()), img_points,
                                                           intrinsic_params, extrinsic_params)
        inside_corner_locations = np.zeros((63, 3))
        for idx in range(63):
            img_points = {}
            for sn in cam_inside_corners.keys():
                if len(cam_inside_corners[sn]):
                    img_points[sn] = cam_inside_corners[sn][idx, :]
            [inside_corner_locations[idx, :], _] = locate(list(img_points.keys()), img_points,
                                                          intrinsic_params, extrinsic_params)

        board.plot_3d(ax, outside_corner_locations, inside_corner_locations)

        if outside_corner_locations.shape[0] > 0:
            xlim = [min(xlim[0], np.min(outside_corner_locations[:, 0])),
                    max(xlim[1], np.max(outside_corner_locations[:, 0]))]
            ylim = [min(ylim[0], np.min(outside_corner_locations[:, 1])),
                    max(ylim[1], np.max(outside_corner_locations[:, 1]))]
            zlim = [min(zlim[0], np.min(outside_corner_locations[:, 2])),
                    max(zlim[1], np.max(outside_corner_locations[:, 2]))]
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

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
        key = cv2.waitKey(1) & 0xFF
        if key == ord('y'):
            accept = True
            break
        elif key == ord('n'):
            accept = False
            break

    return accept


def run_calibration(parameters, run_intrinsic=True, run_extrinsic=True, collect_sba=True):
    """
    Run all calibration
    :param parameters: Acquisition parameters
    :param run_intrinsic: Run instrinsic calibration
    :param run_extrinsic: Run extrinsic calibration
    :param collect_sba: Collect data for SBA
    """
    cams = None
    if parameters['type'] == 'online':
        cams = init_camera_sources(parameters, fps, shutter, gain)

    # Run until final acceptance
    calib_finished = False
    # try:
    while not calib_finished:

        # Intrinsic calibration
        if run_intrinsic:
            intrinsic_params = run_intrinsic_calibration(parameters, cams)
            cv2.destroyAllWindows()
        else:
            handle = open('intrinsic_params.pickle', "rb")
            intrinsic_params = pickle.load(handle)
            handle.close()

        # Extrinsic calibration
        if run_extrinsic:
            extrinsic_params = run_extrinsic_calibration(parameters, cams, intrinsic_params)
            cv2.destroyAllWindows()
        else:
            handle = open('extrinsic_params.pickle', "rb")
            extrinsic_params = pickle.load(handle)
            handle.close()

        # Collect data for SBA
        if collect_sba:
            collect_sba_data(parameters, cams, intrinsic_params, extrinsic_params)
            cv2.destroyAllWindows()

        # Test calibration
        if parameters['type'] == 'online':
            calib_finished = verify_calibration(cams, intrinsic_params, extrinsic_params)
            cv2.destroyAllWindows()
    # except:
    #    pass

    # Close cameras
    for cam in cams:
        cam.close()


if __name__ == '__main__':

    try:
        intrinsic = sys.argv[1] == '1'
        if intrinsic:
            print('RUNNING: intrinsic')
    except:
        intrinsic = True

    try:
        extrinsic = sys.argv[2] == '1'
        if extrinsic:
            print('RUNNING: extrinsic')
    except:
        extrinsic = True

    try:
        sba = sys.argv[3] == '1'
        if sba:
            print('RUNNING: sba')
    except:
        sba = True

    try:
        json_file = sys.argv[4]
        print("USING: ", json_file)
    except:
        json_file = "settings.json"
        print("USING: ", json_file)

    # opening a json file
    with open(json_file) as settings_file:
        params = json.load(settings_file)

    run_calibration(params, run_intrinsic=intrinsic, run_extrinsic=extrinsic, collect_sba=sba)
