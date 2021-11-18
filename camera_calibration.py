import sys

import numpy as np
from ximea import xiapi
import cv2
import pickle
import matplotlib.pyplot as plt
from utilities.calib_tools import triangulate_points, locate

# Rows and columns in chess board
cbcol = 9
cbrow = 6

# Chess board coordinates
objp = np.zeros((cbrow * cbcol, 3), np.float32)
objp[:, :2] = np.mgrid[0:cbcol, 0:cbrow].T.reshape(-1, 2)

subcorner_term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
stereo_term_crit = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 1e-5)
intrinsic_flags = cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_PRINCIPAL_POINT

fps = 60.0
shutter = int((1 / fps) * 1e+6) - 100
gain = 5
f_size = (1280, 1024)

cam_sns = ["06955451","32052251","39050251","32050651"]

def camera_init(sn, framerate, shutter, gain):

    cam = xiapi.Camera()
    cam.open_device_by_SN(sn)  # put a serial number of the camera
    cam.set_exposure(shutter)
    cam.set_gain(gain)
    cam.set_acq_timing_mode("XI_ACQ_TIMING_MODE_FRAME_RATE")
    cam.set_framerate(framerate)
    cam.set_imgdataformat("XI_RGB32")
    cam.enable_auto_wb()
    img = xiapi.Image()

    return cam, img	


def collect_sba_data(cams, imgs, intrinsic_params, extrinsic_params):
    """
    Collect data for sparse bundle adjustment
    :param cam_sns: SN of each camera
    :param cams:  list of camera objects
    :param imgs: image object for each camera
    :param intrinsic_params: intrinsic calibration parameters for each camera
    :param extrinsic_params: extrinsic calibration parameters for each camera
    """
    # Initialize array
    cam_list = {}
    for sn in cam_sns:
        cam_list[sn] = []
    points_3d = []
    point_3d_indices = []
    points_2d = []
    camera_indices = []

    print('Run SBA (y/n)?')

    # Get corresponding points between images
    point_idx_counter = 0
    points_per_image = cbcol * cbrow

    # Acquire enough good frames
    while True:
        # Whether or not chessboard is visible in each camera
        visible_cams = np.zeros((len(cam_sns)))

        # Frames for each camera
        cam_datas = []
        vcam_datas = []
        # Chessboard corners for each camera
        cam_corners = {}

        # Go through each camera
        for cam_idx, (sn, cam, img) in enumerate(zip(cam_sns, cams, imgs)):

            # Get image from camera
            cam.get_image(img)
            cam_data = img.get_image_data_numpy()
            cam_datas.append(cam_data)
            vcam_data = np.copy(cam_data)

            # Convert to greyscale for chess board detection
            gray = cv2.cvtColor(cam_data, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners - fast checking
            ret, corners = cv2.findChessboardCorners(gray, (cbcol, cbrow), None, flags=cv2.CALIB_CB_FAST_CHECK)

            # If found in cameras
            if ret:
                # Refine image points
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), subcorner_term_crit)

                # Visual feedback
                vcam_data = cv2.drawChessboardCorners(vcam_data, (cbcol, cbrow), corners, ret)
                vcam_data = cv2.rectangle(vcam_data, (5, 5), (f_size[0] - 5, f_size[1] - 5), (0, 255, 0),
                                          5)
                visible_cams[cam_idx] = 1
            cam_corners[sn] = corners

            # Num frames
            vcam_data = cv2.putText(vcam_data, '%d frames' % len(cam_list[sn]), (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                    (0, 255, 0), 2)

            # Resize for display
            width = int(f_size[0] * .5)
            height = int(f_size[1] * .5)
            dim = (width, height)
            resized = cv2.resize(vcam_data, dim, interpolation=cv2.INTER_AREA)
            vcam_datas.append(resized)

        # If chessboard visible in more than one camera
        if np.sum(visible_cams) > 1:

            # Which cameras to use for triangulation
            triangulation_cams = []

            # Add 3d and 3d points to list
            for cam_idx, sn in enumerate(cam_sns):
                new_point_3d_indices = range(point_idx_counter, point_idx_counter + points_per_image)
                if visible_cams[cam_idx] > 0:
                    triangulation_cams.append(sn)
                    points_2d.extend(np.array(cam_corners[sn]).reshape(points_per_image, 2))
                    point_3d_indices.extend(new_point_3d_indices)
                    camera_indices.extend([cam_idx] * points_per_image)
                    cam_list[sn].append(cam_datas[cam_idx][:, :, :3])

            # Use the first two cameras to get the initial estimate
            point_3d_est = triangulate_points(triangulation_cams[0], triangulation_cams[1], cam_corners,
                                              intrinsic_params, extrinsic_params)

            points_3d.extend(point_3d_est)
            point_idx_counter += points_per_image

        # Show camera images
        data = np.vstack([np.hstack([vcam_datas[0], vcam_datas[1]]), np.hstack([vcam_datas[2], vcam_datas[3]])])
        cv2.imshow("cam", data)
        k = cv2.waitKey(1) & 0xFF

        # Quit if enough frames
        if k == ord('y'):
            break
        # Start over
        elif k == ord('n'):
            # Initialize array
            cam_list = {}
            for sn in cam_sns:
                cam_list[sn] = []
            points_3d = []
            point_3d_indices = []
            points_2d = []
            camera_indices = []
            point_idx_counter = 0

    # Convert to numpy arrays
    points_2d = np.array(points_2d, dtype=np.float32)
    points_3d = np.array(points_3d, dtype=np.float32)
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
    for sn in cam_sns:
        vid_list = cam_list[sn]
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        cam_vid = cv2.VideoWriter(
            "sba_cam_{}.avi".format(sn),
            fourcc,
            float(fps),
            f_size
        )
        [cam_vid.write(i) for i in vid_list]
        cam_vid.release()


def run_extrinsic_calibration(cams, imgs, intrinsic_params):
    """
    Run extrinsic calibration for each pair of cameras
    :param cam_sns: list of camera SNs
    :param cams: list of camera objects
    :param imgs: image object for each camera
    :param intrinsic_params: intrinsic calibration parameters for each camera
    :return: extrinsic calibration parameters for each camera
    """

    # Rotation matrix for first camera - all others are relative to it
    extrinsic_params = {cam_sns[0]: {}}
    extrinsic_params[cam_sns[0]]['r'] = np.array([[1, 0, 0],
                                                  [0, 0, -1],
                                                  [0, 1, 0]], dtype=np.float32)
    extrinsic_params[cam_sns[0]]['t'] = np.array([[0, 0, 0]], dtype=np.float32).T

    # Go through each camera
    for cam1_idx in range(len(cams)):
        cam1_sn = cam_sns[cam1_idx]
        cam1 = cams[cam1_idx]
        img1 = imgs[cam1_idx]

        # For every other camera (except pairs already calibrated)
        for cam2_idx in range(cam1_idx + 1, len(cams)):
            cam2_sn = cam_sns[cam2_idx]
            cam2 = cams[cam2_idx]
            img2 = imgs[cam2_idx]

            # Calibrate pair
            print("Computing stereo calibration for cam %d and %d" % (cam1_idx, cam2_idx))
            rms, r, t, cam_list = extrinsic_cam_calibration(cam1_sn, cam1, img1, cam2_sn, cam2, img2, intrinsic_params,
                                                            extrinsic_params)
            print(f"{rms:.5f} pixels")

            # https://en.wikipedia.org/wiki/Camera_resectioning#Extrinsic_parameters
            # T is the world origin position in the camera coordinates.
            # The world position of the camera is C = -(R^-1)@T.
            # Similarly, the rotation of the camera in world coordinates is given by R^-1
            extrinsic_params[cam_sns[cam2_idx]] = {}
            extrinsic_params[cam_sns[cam2_idx]]['r'] = r @ extrinsic_params[cam_sns[cam1_idx]]['r']
            extrinsic_params[cam_sns[cam2_idx]]['t'] = r @ extrinsic_params[cam_sns[cam1_idx]]['t'] + t

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


def extrinsic_cam_calibration(sn1, cam1, img1, sn2, cam2, img2, intrinsic_params, extrinsic_params):
    """
    Run extrinsic calibration for one pair of cameras
    :param sn1: SN for camera 1
    :param cam1: camera 1 object
    :param img1: image object for camera 1
    :param sn2: SN for camera 2
    :param cam2: camera 2 object
    :param img2: image object for camera 2
    :param intrinsic_params: intrinsic calibration parameters for each camera
    :param extrinsic_params: extrinsic calibration parameters for each camera
    :return: tuple: RMS, rotation matrix, translation matrix, list of frames for each camera
    """

    # Initialize
    cam_list = {sn1: [], sn2: []}
    objpoints = []
    imgpoints = {sn1: [], sn2: []}
    rms=0
    r = np.array([[1, 0, 0],
                  [0, 0, -1],
                  [0, 1, 0]], dtype=np.float32)
    t = np.array([[0, 0, 0]], dtype=np.float32).T

    # Stop when RMS lower than threshold and greater than min frames collected
    rms_threshold = 1.0
    min_frames = 50
    rms = 1e6
    rmss = []

    # Get intrinsic parameters for each camera
    k1 = intrinsic_params[sn1]['k']
    d1 = intrinsic_params[sn1]['d']
    k2 = intrinsic_params[sn2]['k']
    d2 = intrinsic_params[sn2]['d']

    print('Accept extrinsic calibration (y/n)?')

    # Plot 3D triangulation and RMSE
    fig = plt.figure('extrinsic - %s - %s' % (sn1, sn2))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)

    # Initialize axes limits
    xlim1 = [0, 1]
    ylim1 = [0, 1]
    zlim1 = [0, 1]
    ax1.set_xlim(xlim1[0], xlim1[1])
    ax1.set_ylim(ylim1[0], ylim1[1])
    ax1.set_zlim(zlim1[0], zlim1[1])

    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax2.set_xlabel("frame")
    ax2.set_ylabel("RMS")
    plt.draw()
    plt.pause(0.001)

    # Acquire enough good frames - until RMSE low enough at at least min frames collected
    while rms > rms_threshold or len(imgpoints[sn1]) < min_frames:
        # Get image from cam1
        cam1.get_image(img1)
        cam1_data = img1.get_image_data_numpy()
        vcam1_data = np.copy(cam1_data)

        # Get image from cam2
        cam2.get_image(img2)
        cam2_data = img2.get_image_data_numpy()
        vcam2_data = np.copy(cam2_data)

        # Convert to greyscale for chess board detection
        gray1 = cv2.cvtColor(cam1_data, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(cam2_data, cv2.COLOR_BGR2GRAY)
        img_shape1 = gray1.shape[::-1]

        # Find the chess board corners - fast check
        ret1, corners1 = cv2.findChessboardCorners(gray1, (cbcol, cbrow), None, flags=cv2.CALIB_CB_FAST_CHECK)
        ret2, corners2 = cv2.findChessboardCorners(gray2, (cbcol, cbrow), None, flags=cv2.CALIB_CB_FAST_CHECK)

        # If chessboard found in camera 1
        if ret1:
            # Visual feedback
            vcam1_data = cv2.drawChessboardCorners(vcam1_data, (cbcol, cbrow), corners1, ret1)
            vcam1_data = cv2.rectangle(vcam1_data, (5, 5), (f_size[0] - 5, f_size[1] - 5), (0, 255, 0), 5)

        # If chessboard found in camera 1
        if ret2:
            # Visual feedback
            vcam2_data = cv2.drawChessboardCorners(vcam2_data, (cbcol, cbrow), corners2, ret2)
            vcam2_data = cv2.rectangle(vcam2_data, (5, 5), (f_size[0] - 5, f_size[1] - 5), (0, 255, 0), 5)

        # If found in both cameras, add object points, image points (after refining them)    
        if ret1 and ret2:
            # Add to list of object points
            objpoints.append(objp)

            # Refine image points
            corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), subcorner_term_crit)
            corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), subcorner_term_crit)

            # Add to list of image points            
            imgpoints[sn1].append(corners1)
            imgpoints[sn2].append(corners2)

            # Stereo calibration - keep intrinsic parameters fixed
            rms, *_, r, t, _, _ = cv2.stereoCalibrate(objpoints[-10:], imgpoints[sn1][-10:], imgpoints[sn2][-10:],
                                                      k1, d1, k2, d2, img_shape1, flags=cv2.CALIB_FIX_INTRINSIC,
                                                      criteria=stereo_term_crit)
            # Mean RMSE
            rms = rms / (cbcol * cbrow)
            good_point = True
            # If there is a jump in RMSE - exclude this point
            if len(rmss) > 0 and rms - rmss[-1] > 100:
                objpoints.pop()
                imgpoints[sn1].pop()
                imgpoints[sn2].pop()
                good_point = False

            # Otherwise, update lists and extrinsic parameters
            else:
                rmss.append(rms)

                extrinsic_params[sn2] = {}
                extrinsic_params[sn2]['r'] = r @ extrinsic_params[sn1]['r']
                extrinsic_params[sn2]['t'] = r @ extrinsic_params[sn1]['t'] + t

                cam_list[sn1].append(cam1_data[:, :, :3])
                cam_list[sn2].append(cam2_data[:, :, :3])

            # Triangulate
            ax1.clear()
            [location, pairs_used] = locate([sn1, sn2], {sn1: corners1, sn2: corners2}, intrinsic_params,
                                            extrinsic_params)

            # If seen in both cameras - update plot
            if pairs_used > 0 and good_point:
                xs = location[:, 0]
                ys = location[:, 1]
                zs = location[:, 2]
                ax1.scatter(xs, ys, zs, c='g', marker='o', s=1)
                xlim1 = [min(xlim1[0], np.min(xs)), max(xlim1[1], np.max(xs))]
                ylim1 = [min(ylim1[0], np.min(ys)), max(ylim1[1], np.max(ys))]
                zlim1 = [min(zlim1[0], np.min(zs)), max(zlim1[1], np.max(zs))]

                ax1.set_xlim(xlim1[0], xlim1[1])
                ax1.set_ylim(ylim1[0], ylim1[1])
                ax1.set_zlim(zlim1[0], zlim1[1])
                ax1.set_xlabel("X")
                ax1.set_ylabel("Y")
                ax1.set_zlabel("Z")

        # Plot RMSE
        ax2.clear()
        ax2.plot(range(len(imgpoints[sn1])), rmss)
        ax2.set_xlabel("frame")
        ax2.set_ylabel("RMS")

        plt.draw()
        plt.pause(0.001)

        # Show num frames so far
        vcam1_data = cv2.putText(vcam1_data, '%d frames' % len(imgpoints[sn1]), (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                 (0, 255, 0), 2)
        vcam2_data = cv2.putText(vcam2_data, '%d frames' % len(imgpoints[sn2]), (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                 (0, 255, 0), 2)

        # Resize for display
        width = int(f_size[0] * .5)
        height = int(f_size[1] * .5)
        dim = (width, height)
        resized1 = cv2.resize(vcam1_data, dim, interpolation=cv2.INTER_AREA)

        # Resize for display
        width = int(f_size[0] * .5)
        height = int(f_size[1] * .5)
        dim = (width, height)
        resized2 = cv2.resize(vcam2_data, dim, interpolation=cv2.INTER_AREA)

        # Show camera images side by side
        data1 = np.hstack([resized1, resized2])
        cv2.imshow("cam", data1)
        k = cv2.waitKey(1) & 0xFF

        # Quit if enough frames
        if k == ord('y'):
            break
        # Start over
        elif k == ord('n'):
            cam_list = {sn1: [], sn2: []}
            objpoints = []
            imgpoints = {sn1: [], sn2: []}
            rmss = []
            xlim1 = [0, 1]
            ylim1 = [0, 1]
            zlim1 = [0, 1]

    if len(objpoints)>0:
        # Final stereo calibration - keep intrinsic parameters fixed
        (rms, *_, r, t, _, _) = cv2.stereoCalibrate(objpoints, imgpoints[sn1], imgpoints[sn2],
                                                    k1, d1, k2, d2, img_shape1, flags=cv2.CALIB_FIX_INTRINSIC,
                                                    criteria=stereo_term_crit)
        # Mean RMSE
        rms = rms / (cbcol * cbrow)

    return rms, r, t, cam_list


def run_intrinsic_calibration(cams, imgs):
    """
    Run intrinsic calibration for each camera
    :param cam_sns: list of camera SNs
    :param cams: list of camera objects
    :param imgs: image object for each camera
    :return: intrinsic calibration parameters for each camera
    """
    intrinsic_params = {}

    # Calibrate each camera
    for sn, cam, img in zip(cam_sns, cams, imgs):
        print('Calibrating camera %s' % sn)
        rpe, k, d, cam_list = intrinsic_cam_calibration(sn, cam, img)
        print("Mean re-projection error: %.3f pixels " % rpe)
        intrinsic_params[sn] = {
            'k': k,
            'd': d
        }

        # Save frames for calibration
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        cam_vid = cv2.VideoWriter(
            "intrinsic_cam_{}.avi".format(sn),
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


def intrinsic_cam_calibration(sn, cam, img):
    """
    Run intrinsic calibration for one camera
    :param sn: camera SN
    :param cam: camera object
    :param img: image object for camera
    :return: tuple: reprojection error, k, distortion coefficient, list of frames for each camera
    """
    cam_list = []
    objpoints = []
    imgpoints = []

    # Stop when RPE lower than threshold and greater than min frames collected
    rpe_threshold = 0.5
    min_frames = 50
    rpe = 1e6
    rpes = []

    # Initialize params
    k = np.zeros((3, 3))
    d = np.zeros((1, 5))
    map_x, map_y = cv2.initUndistortRectifyMap(k, d, None, k, (f_size[0], f_size[1]), cv2.CV_32FC1)

    # Plot RPE
    fig = plt.figure('Intrinsic - %s' % sn)
    ax = fig.add_subplot(111)
    ax.set_xlabel("frame")
    ax.set_ylabel("RPE")
    plt.draw()
    plt.pause(0.001)

    print('Accept intrinsic calibration (y/n)?')

    # Acquire enough good frames - until RPE low enough at at least min frames collected
    while rpe > rpe_threshold or len(imgpoints) < min_frames:

        # Get image from camera
        cam.get_image(img)
        cam_data = img.get_image_data_numpy()
        undist_data = cv2.remap(cam_data, map_x, map_y, cv2.INTER_LINEAR)
        vcam_data = np.copy(cam_data)
        vundist_data = np.copy(undist_data)

        # Convert to greyscale for chess board detection
        gray = cv2.cvtColor(cam_data, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners - fast check
        ret, corners = cv2.findChessboardCorners(gray, (cbcol, cbrow), None, flags=cv2.CALIB_CB_FAST_CHECK)

        # If found, add object points, image points (after refining them)   
        if ret:
            img_shape = gray.shape[::-1]
            objpoints.append(objp)

            # Refine image points
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), subcorner_term_crit)
            imgpoints.append(corners)

            # Visual feedback
            vcam_data = cv2.drawChessboardCorners(vcam_data, (cbcol, cbrow), corners, ret)
            vcam_data = cv2.rectangle(vcam_data, (5, 5), (f_size[0] - 5, f_size[1] - 5), (0, 255, 0), 5)

            # Intrinsic calibration
            rpe, k, d, r, t = cv2.calibrateCamera(objpoints[-10:], imgpoints[-10:], img_shape, None, None, flags=intrinsic_flags)

            # Undistort
            undist_corners = cv2.undistortPoints(corners, k, d)
            vundist_data = cv2.drawChessboardCorners(vundist_data, (cbcol, cbrow), undist_corners, ret)
            vundist_data = cv2.rectangle(vundist_data, (5, 5), (f_size[0] - 5, f_size[1] - 5), (0, 255, 0), 5)

            # If there is a jump in RPE - exclude this point
            if len(rpes) > 0 and rpe - rpes[-1] > 100:
                objpoints.pop()
                imgpoints.pop()
            # Otherwise, update lists and undistort rectify map
            else:
                rpes.append(rpe)
                map_x, map_y = cv2.initUndistortRectifyMap(k, d, None, k, img_shape, cv2.CV_32FC1)
                cam_list.append(cam_data[:, :, :3])

        # Number of frames so far
        vcam_data = cv2.putText(vcam_data, '%d frames' % len(imgpoints), (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                (0, 255, 0), 2)

        # Plot projection error
        ax.clear()
        ax.plot(range(len(imgpoints)), rpes)
        ax.set_xlabel("frame")
        ax.set_ylabel("RPE")
        plt.draw()
        plt.pause(0.001)

        # Resize for display    
        width = int(f_size[0] * .5)
        height = int(f_size[1] * .5)
        dim = (width, height)
        resized = cv2.resize(vcam_data, dim, interpolation=cv2.INTER_AREA)

        # Resize for display
        width = int(f_size[0] * .5)
        height = int(f_size[1] * .5)
        dim = (width, height)
        undist_resized = cv2.resize(vundist_data, dim, interpolation=cv2.INTER_AREA)

        # Show camera image and undistorted image side by side
        data = np.hstack([resized, undist_resized])
        cv2.imshow("cam", data)
        k = cv2.waitKey(1) & 0xFF

        # Quit if enough frames
        if k == ord('y'):
            break
        # Start over
        elif k == ord('n'):
            cam_list = []
            objpoints = []
            imgpoints = []
            rpes = []

    # Final camera calibration
    rpe, k, d, r, t = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None, flags=intrinsic_flags)

    # Mean RPE
    rpe = rpe / (cbcol * cbrow)

    return rpe, k, d, cam_list


def verify_calibration(cams, imgs, intrinsic_params, extrinsic_params):
    """
    Verify calibration
    :param cam_sns: list of camera SNs
    :param cams: list of camera objects
    :param imgs: image object for each camera
    :param intrinsic_params: intrinsic calibration parameters for each camera
    :param extrinsic_params: extrinsic calibration parameters for each camera
    :return: whether or not to accept calibration
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
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
    print('Accept final calibration (y/n)?')

    while True:
        cam_datas = []
        cam_coords = {}

        for sn, cam, img in zip(cam_sns, cams, imgs):
            cam.get_image(img)

            cam_data = img.get_image_data_numpy()

            gray = cv2.cvtColor(cam_data, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (cbcol, cbrow), None, flags=cv2.CALIB_CB_FAST_CHECK)

            cam_coords[sn] = []
            if ret:
                cam_coords[sn] = corners
                cam_data = cv2.drawChessboardCorners(cam_data, (cbcol, cbrow), corners, ret)
                cam_data = cv2.rectangle(cam_data, (5, 5), (f_size[0] - 5, f_size[1] - 5), (0, 255, 0), 5)

            width = int(f_size[0] * .5)
            height = int(f_size[1] * .5)
            dim = (width, height)
            resized = cv2.resize(cam_data, dim, interpolation=cv2.INTER_AREA)
            cam_datas.append(resized)

        ax.clear()
        [location, pairs_used] = locate(cam_sns, cam_coords, intrinsic_params, extrinsic_params)
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

        data = np.vstack([np.hstack([cam_datas[0], cam_datas[1]]), np.hstack([cam_datas[2], cam_datas[3]])])

        cv2.imshow("cam", data)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('y'):
            accept = True
            break
        elif k == ord('n'):
            accept = False
            break
    plt.close('all')
    plt.draw()
    plt.pause(0.001)
    cv2.destroyAllWindows()
    return accept


def run_calibration(run_intrinsic=True, run_extrinsic=True, collect_sba=True):
    """
    Run all calibration
    """
    cams = []
    imgs = []

    ############################
    # for each camera separately
    for sn in cam_sns:
        cam, img = camera_init(sn, fps, shutter, gain)
		
        cam.start_acquisition()
        cams.append(cam)
        imgs.append(img)

    # Run until final acceptance
    calib_finished = False
    while not calib_finished:

        # Intrinsic calibration
        if run_intrinsic:
            intrinsic_params = run_intrinsic_calibration(cams, imgs)
            cv2.destroyAllWindows()
        else:
            handle = open('intrinsic_params.pickle', "rb")
            intrinsic_params = pickle.load(handle)
            handle.close()

        # Extrinsic calibration
        if run_extrinsic:
            extrinsic_params = run_extrinsic_calibration(cams, imgs, intrinsic_params)
            cv2.destroyAllWindows()
        else:
            handle = open('extrinsic_params.pickle', "rb")
            extrinsic_params = pickle.load(handle)
            handle.close()

        # Collect data for SBA
        if collect_sba:
            collect_sba_data(cams, imgs, intrinsic_params, extrinsic_params)
            cv2.destroyAllWindows()

        # Test calibration
        calib_finished = verify_calibration(cams, imgs, intrinsic_params, extrinsic_params)

    # Close cameras
    for cam in cams:
        cam.stop_acquisition()
        cam.close_device()


if __name__ == '__main__':
    intrinsic = True
    extrinsic = True
    sba = True
    if len(sys.argv) > 1:
        intrinsic = sys.argv[1] == '1'
    if len(sys.argv) > 2:
        extrinsic = sys.argv[2] == '1'
    if len(sys.argv) > 3:
        sba = sys.argv[3] == '1'
    run_calibration(run_intrinsic=intrinsic, run_extrinsic=extrinsic, collect_sba=sba)
