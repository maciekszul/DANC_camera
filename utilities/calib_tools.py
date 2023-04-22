import json
import os
import pickle
import sys
from time import time

import matplotlib
import numpy as np
import cv2
import yaml as yaml
from cv2 import aruco
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
from scipy import linalg
import matplotlib.pyplot as plt


def project_points(obj_pts, k, d, r, t):
    pts = cv2.projectPoints(obj_pts, r, t, k, d)[0].reshape((-1, 2))
    return pts


def params_to_points_extrinsics(params, n_cams, n_points):
    r_end_idx = n_cams * 3
    t_end_idx = r_end_idx + n_cams * 3
    r_vecs = params[:r_end_idx].reshape((n_cams, 3))
    r_arr = np.array([cv2.Rodrigues(r)[0] for r in r_vecs], dtype=np.float64)
    t_arr = params[r_end_idx:t_end_idx].reshape((n_cams, 3, 1))
    obj_pts = params[t_end_idx:].reshape((n_points, 3))
    return obj_pts, r_arr, t_arr


def cost_func_points_extrinsics(params, n_cams, n_points, point_3d_indices, camera_indices, k_arr, d_arr, points_2d):
    obj_pts, r_arr, t_arr = params_to_points_extrinsics(params, n_cams, n_points)
    reprojected_pts = np.array([project_points(obj_pts[i], k_arr[j], d_arr[j], r_arr[j], t_arr[j])[0] for i, j in
                                zip(point_3d_indices, camera_indices)])
    error = (reprojected_pts - points_2d).ravel()
    return error


def cost_func_points_only(params, n_points, point_3d_indices, camera_indices, k_arr, d_arr, r_arr, t_arr, points_2d):
    obj_pts = params.reshape((n_points, 3))
    reprojected_pts = np.array([project_points(obj_pts[i], k_arr[j], d_arr[j], r_arr[j], t_arr[j])[0] for i, j in
                                zip(point_3d_indices, camera_indices)])
    error = (reprojected_pts - points_2d).ravel()
    return error


def create_bundle_adjustment_jacobian_sparsity_matrix(n_cams, n_params_per_camera, camera_indices, n_points,
                                                      point_indices):
    m = camera_indices.size * 2
    n = n_cams * n_params_per_camera + n_points * 3
    A = lil_matrix((m, n), dtype=int)
    i = np.arange(camera_indices.size)
    for s in range(n_params_per_camera):
        A[2 * i, camera_indices * n_params_per_camera + s] = 1
        A[2 * i + 1, camera_indices * n_params_per_camera + s] = 1
    for s in range(3):
        A[2 * i, n_cams * n_params_per_camera + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cams * n_params_per_camera + point_indices * 3 + s] = 1
    return A


def bundle_adjust_points_and_extrinsics(points_2d, points_3d, point_3d_indices, camera_indices, k_arr, d_arr, r_arr,
                                        t_arr):
    n_points = len(points_3d)
    n_cams = len(k_arr)
    n_params_per_camera = 6
    r_vecs = np.array([cv2.Rodrigues(r)[0] for r in r_arr], dtype=np.float64).flatten()
    t_vecs = t_arr.flatten()
    x0 = np.concatenate([r_vecs, t_vecs, points_3d.flatten()])
    f0 = cost_func_points_extrinsics(x0, n_cams, n_points, point_3d_indices, camera_indices, k_arr, d_arr, points_2d)
    A = create_bundle_adjustment_jacobian_sparsity_matrix(n_cams, n_params_per_camera, camera_indices, n_points,
                                                          point_3d_indices)
    t0 = time()
    res = least_squares(cost_func_points_extrinsics, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-10,
                        method='trf', loss='cauchy',
                        args=(n_cams, n_points, point_3d_indices, camera_indices, k_arr, d_arr, points_2d),
                        max_nfev=1000)
    t1 = time()
    print("\nOptimization took {0:.2f} seconds".format(t1 - t0))
    obj_pts, r_arr, t_arr = params_to_points_extrinsics(res.x, n_cams, n_points)
    residuals = dict(before=f0, after=res.fun)
    return obj_pts, r_arr, t_arr, residuals


def bundle_adjust_points_only(points_2d, points_3d, point_3d_indices, camera_indices, k_arr, d_arr, r_arr, t_arr,
                              f_scale=50):
    n_points = len(points_3d)
    n_cams = len(k_arr)
    n_params_per_camera = 0
    x0 = points_3d.flatten()
    f0 = cost_func_points_only(x0, n_points, point_3d_indices, camera_indices, k_arr, d_arr, r_arr, t_arr, points_2d)
    A = create_bundle_adjustment_jacobian_sparsity_matrix(n_cams, n_params_per_camera, camera_indices, n_points,
                                                          point_3d_indices)
    t0 = time()
    res = least_squares(cost_func_points_only, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-15, method='trf',
                        loss='cauchy', f_scale=f_scale,
                        args=(n_points, point_3d_indices, camera_indices, k_arr, d_arr, r_arr, t_arr, points_2d),
                        max_nfev=500)
    t1 = time()
    print("\nOptimization took {0:.2f} seconds".format(t1 - t0))
    residuals = dict(before=f0, after=res.fun)
    obj_pts = res.x.reshape((n_points, 3))
    return obj_pts, residuals


def triangulate_points(sn1, sn2, img_pts, intrinsic_params, extrinsic_params):
    img_pts_1 = img_pts[sn1]
    img_pts_2 = img_pts[sn2]
    k1 = intrinsic_params[sn1]['k']
    d1 = intrinsic_params[sn1]['d']
    r1 = extrinsic_params[sn1]['r']
    t1 = extrinsic_params[sn1]['t']
    k2 = intrinsic_params[sn2]['k']
    d2 = intrinsic_params[sn2]['d']
    r2 = extrinsic_params[sn2]['r']
    t2 = extrinsic_params[sn2]['t']
    pts_1 = img_pts_1.reshape((-1, 1, 2))
    pts_2 = img_pts_2.reshape((-1, 1, 2))
    pts_1 = cv2.undistortPoints(pts_1, k1, d1)
    pts_2 = cv2.undistortPoints(pts_2, k2, d2)
    p1 = np.hstack((r1, t1))
    p2 = np.hstack((r2, t2))
    pts_4d = cv2.triangulatePoints(p1, p2, pts_1, pts_2)
    pts_3d = (pts_4d[:3] / pts_4d[3]).T
    return pts_3d

def rectify_coord(coord, rectify_params):
    table_center = rectify_params['origin']
    v1 = rectify_params['x_axis'] - table_center
    v2 = rectify_params['y_axis'] - table_center
    v3 = rectify_params['z_axis'] - table_center
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    v3 = v3 / np.linalg.norm(v3)
    M_inv = np.linalg.inv(np.transpose(np.squeeze([v1, v2, v3])))
    coord = np.transpose(np.matmul(M_inv, np.transpose((coord - table_center))))
    return coord

def locate(cam_sns, camera_coords, intrinsic_params, extrinsic_params, rectify_params=None):
    location = np.zeros((1, 3))
    pairs_used = 0
    for idx1 in range(len(cam_sns)):
        sn1 = cam_sns[idx1]
        for idx2 in range(idx1 + 1, len(cam_sns)):
            sn2 = cam_sns[idx2]
            if sn1 in camera_coords and len(camera_coords[sn1]) > 0 and sn2 in camera_coords and len(
                    camera_coords[sn2]) > 0:
                img_pts = {
                    sn1: camera_coords[sn1],
                    sn2: camera_coords[sn2]
                }
                result = triangulate_points(sn1, sn2, img_pts, intrinsic_params, extrinsic_params)
                location = location + result
                pairs_used = pairs_used + 1
    if pairs_used > 0:
        location = location / pairs_used
    # Apply rectification
    if rectify_params is not None:
        location = rectify_coord(location, rectify_params)
    return [location, pairs_used]


def locate_dlt(cam_sns, camera_coords, intrinsic_params, extrinsic_params, rectify_params=None):
    A = []
    cameras_used = 0
    location = np.zeros((1, 3))
    for idx in range(len(cam_sns)):
        sn = cam_sns[idx]
        if len(camera_coords[sn]) > 0:
            point = camera_coords[sn]
            RT = np.concatenate([extrinsic_params[sn]['r'], extrinsic_params[sn]['t']], axis=-1)
            P = intrinsic_params[sn]['k'] @ RT
            A.append(point[1] * P[2, :] - P[1, :])
            A.append(P[0, :] - point[0] * P[2, :])
            cameras_used = cameras_used + 1
    if cameras_used > 1:
        A = np.array(A)
        A = np.array(A).reshape((cameras_used * 2, 4))
        # print('A: ')
        # print(A)

        B = A.transpose() @ A
        U, s, Vh = linalg.svd(B, full_matrices=False)

        location = Vh[-1, 0:3] / Vh[-1, 3]
        location = np.reshape(location, (1, 3))

        # Apply rectification
        if rectify_params is not None:
            location = rectify_coord(location, rectify_params)
    return [location, cameras_used]


def locate_sba(cam_sns, camera_coords, intrinsic_params, extrinsic_params, rectify_params=None):
    k_arr = np.array([intrinsic_params[x]['k'] for x in cam_sns])
    d_arr = np.array([intrinsic_params[x]['d'] for x in cam_sns])
    r_arr = np.array([extrinsic_params[x]['r'] for x in cam_sns])
    t_arr = np.array([extrinsic_params[x]['t'] for x in cam_sns])

    location = np.zeros((1, 3))
    pts_3d = np.zeros((1, 3))
    points_2d = []
    point_indices = []
    camera_indices = []

    pairs_used = 0
    for idx1 in range(len(cam_sns)):
        sn1 = cam_sns[idx1]
        for idx2 in range(idx1 + 1, len(cam_sns)):
            sn2 = cam_sns[idx2]
            img_pts = {
                sn1: camera_coords[sn1],
                sn2: camera_coords[sn2]
            }

            if len(camera_coords[sn1]) > 0 and len(camera_coords[sn2]) > 0:
                points_2d.extend(np.array(camera_coords[sn1]).reshape(1, 2))
                point_indices.append(0)
                camera_indices.append(idx1)

                points_2d.extend(np.array(camera_coords[sn2]).reshape(1, 2))
                point_indices.append(0)
                camera_indices.append(idx2)

                result = triangulate_points(sn1, sn2, img_pts, intrinsic_params, extrinsic_params)
                location = location + result
                pairs_used = pairs_used + 1

    if pairs_used > 0:
        location = location / pairs_used

        print('bundle_adjust_points_only')

        points_2d = np.array(points_2d, dtype=np.float32)
        points_3d = np.array(location, dtype=np.float32)
        point_indices = np.array(point_indices, dtype=np.int)
        camera_indices = np.array(camera_indices, dtype=np.int)

        pts_3d, res = bundle_adjust_points_only(points_2d, points_3d, point_indices, camera_indices, k_arr, d_arr,
                                                r_arr, t_arr, f_scale=50)
        print(f"\nBefore: mean: {np.mean(res['before'])}, std: {np.std(res['before'])}")
        print(f"After: mean: {np.mean(res['after'])}, std: {np.std(res['after'])}\n")

    # Apply rectification
    if rectify_params is not None:
        pts_3d = rectify_coord(pts_3d, rectify_params)

    return [pts_3d, pairs_used]


class DoubleCharucoBoard:
    def __init__(self):
        self.n_squares_width = 10  # number of squares width
        self.n_squares_height = 7  # number of squares height
        self.n_square_corners = (self.n_squares_height-1)*(self.n_squares_width-1)

        self.square_length = 0.03 # 30mm
        self.marker_length = 0.02 # 20mm

        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        self.board1 = cv2.aruco.CharucoBoard_create(self.n_squares_width, self.n_squares_height, self.square_length,
                                                    self.marker_length, self.dictionary)
        self.board2 = cv2.aruco.CharucoBoard_create(self.n_squares_width, self.n_squares_height, self.square_length,
                                                    self.marker_length, self.dictionary)
        self.board2.ids = self.board2.ids + len(self.board1.ids)

        pixels_per_mm = 6
        pix_w = int(self.n_squares_width * self.square_length * 1000 * pixels_per_mm)
        pix_h = int(self.n_squares_height * self.square_length * 1000 * pixels_per_mm)
        self.imboard1 = self.board1.draw((pix_w, pix_h))
        self.imboard2 = self.board2.draw((pix_w, pix_h))

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(2, 1, 1)
        plt.imshow(self.imboard1, cmap=matplotlib.cm.gray, interpolation="nearest")
        ax.axis("off")
        ax = fig.add_subplot(2, 1, 2)
        plt.imshow(self.imboard2, cmap=matplotlib.cm.gray, interpolation="nearest")
        ax.axis("off")
        plt.show()

    def save(self):
        cv2.imwrite("../charuco_board_1.png", self.imboard1)
        cv2.imwrite("../charuco_board_2.png", self.imboard2)

    def project(self, board, k, d, rvec, tvec):
        outside_corners = [
            [np.min(board.chessboardCorners[:, 0]), np.min(board.chessboardCorners[:, 1]), 0],
            [np.max(board.chessboardCorners[:, 0]), np.min(board.chessboardCorners[:, 1]), 0],
            [np.min(board.chessboardCorners[:, 0]), np.max(board.chessboardCorners[:, 1]), 0],
            [np.max(board.chessboardCorners[:, 0]), np.max(board.chessboardCorners[:, 1]), 0]]
        outside_corners = np.array(outside_corners)
        [rmat, _] = cv2.Rodrigues(rvec)
        outside_projected = np.zeros((outside_corners.shape[0], 2))
        for i in range(outside_corners.shape[0]):
            ptProj = np.matmul(rmat, np.reshape(outside_corners[i, :], (3, 1))) + tvec
            [outside_projected[i, :], _] = cv2.projectPoints(ptProj, (0, 0, 0), (0, 0, 0), k, d)

        inside_corners = board.chessboardCorners
        [rmat, _] = cv2.Rodrigues(rvec)
        inside_projected = np.zeros((inside_corners.shape[0], 2))
        for i in range(inside_corners.shape[0]):
            ptProj = np.matmul(rmat, np.reshape(inside_corners[i, :], (3, 1))) + tvec
            [inside_projected[i, :], _] = cv2.projectPoints(ptProj, (0, 0, 0), (0, 0, 0), k, d)

        return outside_projected, inside_projected

    def get_detected_board(self, marker_ids):
        if marker_ids is not None and len(marker_ids)>0:
            if len(np.where(self.board1.ids == marker_ids[0])[0]) > 0:
                return self.board1
            elif len(np.where(self.board2.ids == marker_ids[0])[0]) > 0:
                return self.board2
        return None

    #
    # Get the id of the corner in the other board
    #
    # def get_corresponding_corner_id(self, corner_id, cam_board):
    #
    #     # This is the order of the corners in the second board
    #     #board2_matching_ids = np.reshape(np.flip(np.reshape(np.array(range(self.n_square_corners)), ((self.n_squares_height-1), (self.n_squares_width-1))), 0), (self.n_square_corners,))
    #
    #     board2_matching_ids=np.array(range(self.n_square_corners))
    #     board2_corner_id=board2_matching_ids[corner_id]
    #     if cam_board is not None:
    #         if cam_board.ids[0] == 0:
    #             return corner_id
    #         else:
    #             return board2_corner_id
    #     return None

    def get_corresponding_corner_id(self, corner_id, detected_board):

        if detected_board is not None:
            if detected_board.ids[0] == 0:
                return corner_id
            else:
                num_cols = self.n_squares_width - 1
                row = corner_id // num_cols
                col = corner_id % num_cols
                return row * num_cols + (num_cols - 1 - col)
        return None

    @staticmethod
    def plot_3d(ax, outside_corner_locations, inside_corner_locations):
        if outside_corner_locations.shape[0] > 0:
            ax.plot([outside_corner_locations[0, 0], outside_corner_locations[1, 0]],
                    [outside_corner_locations[0, 1], outside_corner_locations[1, 1]],
                    zs=[outside_corner_locations[0, 2], outside_corner_locations[1, 2]])
            ax.plot([outside_corner_locations[1, 0], outside_corner_locations[3, 0]],
                    [outside_corner_locations[1, 1], outside_corner_locations[3, 1]],
                    zs=[outside_corner_locations[1, 2], outside_corner_locations[3, 2]])
            ax.plot([outside_corner_locations[2, 0], outside_corner_locations[3, 0]],
                    [outside_corner_locations[2, 1], outside_corner_locations[3, 1]],
                    zs=[outside_corner_locations[2, 2], outside_corner_locations[3, 2]])
            ax.plot([outside_corner_locations[2, 0], outside_corner_locations[0, 0]],
                    [outside_corner_locations[2, 1], outside_corner_locations[0, 1]],
                    zs=[outside_corner_locations[2, 2], outside_corner_locations[0, 2]])

            cmap = plt.get_cmap('viridis')
            ncolors = len(cmap.colors)
            ncorners = inside_corner_locations.shape[0]
            for idx in range(ncorners):
                col_idx = int((idx / ncorners) * ncolors)
                ax.scatter(inside_corner_locations[idx, 0], inside_corner_locations[idx, 1],
                           inside_corner_locations[idx, 2],
                           color=cmap.colors[col_idx], marker='o', s=1)


class ArucoCube:
    def __init__(self):
        # Cube is 40x40mm
        self.cube_width = 0.040
        # Markers are 30x30mm
        self.marker_width = 0.030
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)
        self.board_ids = np.array([[0], [5], [3], [2], [1], [4]], dtype=np.int32)
        self.board_corners = [
            np.array([[-.5*self.marker_width, .5*self.marker_width, .5*self.cube_width],
                      [.5*self.marker_width, .5*self.marker_width, .5*self.cube_width],
                      [.5*self.marker_width, -.5*self.marker_width, .5*self.cube_width],
                      [-.5*self.marker_width, -.5*self.marker_width, .5*self.cube_width]],
                     dtype=np.float32),
            np.array([[-.5*self.marker_width, -.5*self.cube_width, .5*self.marker_width],
                      [.5*self.marker_width, -.5*self.cube_width, .5*self.marker_width],
                      [.5*self.marker_width, -.5*self.cube_width, -.5*self.marker_width],
                      [-.5*self.marker_width, -.5*self.cube_width, -.5*self.marker_width]],
                     dtype=np.float32),
            np.array([[-.5*self.cube_width, .5*self.marker_width, -.5*self.marker_width],
                      [-.5*self.cube_width, .5*self.marker_width, .5*self.marker_width],
                      [-.5*self.cube_width, -.5*self.marker_width, .5*self.marker_width],
                      [-.5*self.cube_width, -.5*self.marker_width, -.5*self.marker_width]],
                     dtype=np.float32),
            np.array([[.5*self.marker_width, .5*self.marker_width, -.5*self.cube_width],
                      [-.5*self.marker_width, .5*self.marker_width, -.5*self.cube_width],
                      [-.5*self.marker_width, -.5*self.marker_width, -.5*self.cube_width],
                      [.5*self.marker_width, -.5*self.marker_width, -.5*self.cube_width]],
                     dtype=np.float32),
            np.array([[.5*self.cube_width, .5*self.marker_width, .5*self.marker_width],
                      [.5*self.cube_width, .5*self.marker_width, -.5*self.marker_width],
                      [.5*self.cube_width, -.5*self.marker_width, -.5*self.marker_width],
                      [.5*self.cube_width, -.5*self.marker_width, .5*self.marker_width]],
                     dtype=np.float32),
            np.array([[-.5*self.marker_width, .5*self.cube_width, -.5*self.marker_width],
                      [.5*self.marker_width, .5*self.cube_width, -.5*self.marker_width],
                      [.5*self.marker_width, .5*self.cube_width, .5*self.marker_width],
                      [-.5*self.marker_width, .5*self.cube_width, .5*self.marker_width]],
                     dtype=np.float32)
        ]
        self.board = aruco.Board_create(self.board_corners, self.aruco_dict, self.board_ids)
        self.imboard = self.draw_aruco_board()

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.imshow(self.imboard, cmap=matplotlib.cm.gray, interpolation="nearest")
        ax.axis("off")
        plt.show()

    def save(self):
        cv2.imwrite("../aruco_cube.png", self.imboard)

    def draw_aruco_board(self):
        paper_px_width = 300
        paper_width = 0.2159  # This value is in meters, for 8.5x11" paper
        face_width = 0.06  # This value is in meters
        cube_template = np.ones((int(paper_px_width * 11 / 8.5), paper_px_width)) * 255
        marker_px = int((self.marker_width / paper_width) * cube_template.shape[1] / 2)
        face_px = int((face_width / paper_width) * cube_template.shape[1] / 2)

        def drawMarkerAt(id, marker_center):
            marker = aruco.drawMarker(self.aruco_dict, id, marker_px * 2, borderBits=1)
            padded_marker = np.ones((face_px * 2, face_px * 2)) * 255
            padding = face_px - marker_px
            padded_marker[padding:(face_px * 2) - padding, padding:(face_px * 2) - padding] = marker
            cv2.rectangle(padded_marker, (0, 0), (padded_marker.shape[0] - 1, padded_marker.shape[1] - 1), 0, 1)
            cube_template[marker_center[1] - face_px:marker_center[1] + face_px, marker_center[0] - face_px:marker_center[0] + face_px] = padded_marker

        drawMarkerAt(self.board_ids[0][0], (int(cube_template.shape[1] / 2), face_px))
        drawMarkerAt(self.board_ids[1][0], (int(cube_template.shape[1] / 2), face_px + 1 * face_px * 2))
        drawMarkerAt(self.board_ids[2][0], (int(cube_template.shape[1] / 2) - face_px * 2, face_px + 2 * face_px * 2))
        drawMarkerAt(self.board_ids[3][0], (int(cube_template.shape[1] / 2), face_px + 2 * face_px * 2))
        drawMarkerAt(self.board_ids[4][0], (int(cube_template.shape[1] / 2) + face_px * 2, face_px + 2 * face_px * 2))
        drawMarkerAt(self.board_ids[5][0], (int(cube_template.shape[1] / 2), face_px + 3 * face_px * 2))
        return cube_template

    def project(self, k, d, rvec, tvec):
        [rmat, _] = cv2.Rodrigues(rvec)
        origin = np.zeros((3, 1))
        pt_origin = np.matmul(rmat, origin) + tvec
        [origin_projected, _] = cv2.projectPoints(pt_origin, (0, 0, 0), (0, 0, 0), k, d)
        origin = np.squeeze(origin_projected)

        x_axis = np.array([[self.marker_width], [0], [0]])
        pt_x = np.matmul(rmat, x_axis) + tvec
        [x_projected, _] = cv2.projectPoints(pt_x, (0, 0, 0), (0, 0, 0), k, d)
        x_axis = np.squeeze(x_projected)

        y_axis = np.array([[0], [self.marker_width], [0]])
        pt_y = np.matmul(rmat, y_axis) + tvec
        [y_projected, _] = cv2.projectPoints(pt_y, (0, 0, 0), (0, 0, 0), k, d)
        y_axis = np.squeeze(y_projected)

        z_axis = np.array([[0], [0], [self.marker_width]])
        pt_z = np.matmul(rmat, z_axis) + tvec
        [z_projected, _] = cv2.projectPoints(pt_z, (0, 0, 0), (0, 0, 0), k, d)
        z_axis = np.squeeze(z_projected)

        return origin, x_axis, y_axis, z_axis

    @staticmethod
    def plot_3d(ax, origin_location, x_location, y_location, z_location):
        ax.scatter(origin_location[:, 0], origin_location[:, 1], origin_location[:, 2], c='k', marker='o', s=1)

        ax.plot([origin_location[0, 0], x_location[0, 0]], [origin_location[0, 1], x_location[0, 1]],
                [origin_location[0, 2], x_location[0, 2]], c='r')

        ax.plot([origin_location[0, 0], y_location[0, 0]], [origin_location[0, 1], y_location[0, 1]],
                [origin_location[0, 2], y_location[0, 2]], c='g')

        ax.plot([origin_location[0, 0], z_location[0, 0]], [origin_location[0, 1], z_location[0, 1]],
                [origin_location[0, 2], z_location[0, 2]], c='b')


# Convert calibration to Jarvis format
def convert_calibration_to_jarvis(calibration_path):

    json_file= os.path.join(calibration_path, "settings.json")
    with open(json_file) as settings_file:
        params = json.load(settings_file)

    # Load intrinsic parameters
    handle = open(os.path.join(calibration_path, "intrinsic_params.pickle"), 'rb')
    intrinsic_params = pickle.load(handle)
    handle.close()

    # Load extrinsic parameters
    handle = open(os.path.join(calibration_path, "extrinsic_params.pickle"), 'rb')
    extrinsic_params = pickle.load(handle)
    handle.close()

    # For each camera in params
    for cam_sn in params["cam_sns"]:
        # yaml file to write to
        yaml_file = os.path.join(calibration_path, "cam{}.yaml".format(cam_sn))
        intrinsic_matrix = intrinsic_params[cam_sn]["k"]
        distortion_coefficients = intrinsic_params[cam_sn]["d"]
        rotation_matrix = extrinsic_params[cam_sn]["r"]
        translation_vector = extrinsic_params[cam_sn]["t"]
        yaml_data={
            'intrinsicMatrix': intrinsic_matrix.astype(np.float64).T,
            'distortionCoefficients': distortion_coefficients.astype(np.float64),
            'R': rotation_matrix.astype(np.float64).T,
            'T': translation_vector.astype(np.float64)*1000
        }

        cv_file = cv2.FileStorage(yaml_file, cv2.FILE_STORAGE_WRITE)
        for key in yaml_data:
            cv_file.write(key, yaml_data[key])
        # Note you *release*; you don't close() a FileStorage object
        cv_file.release()


if __name__ == '__main__':
    convert_calibration_to_jarvis(sys.argv[1])
