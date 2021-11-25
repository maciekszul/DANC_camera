import copy
from time import time

import matplotlib
import numpy as np
import cv2
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


def locate(cam_sns, camera_coords, intrinsic_params, extrinsic_params):
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
    return [location, pairs_used]


def locate_dlt(cam_sns, camera_coords, intrinsic_params, extrinsic_params):
    A = []
    cameras_used = 0
    location = np.zeros((1, 3))
    for idx in range(len(cam_sns)):
        sn = cam_sns[idx]
        if len(camera_coords[sn]) > 0:
            point = camera_coords[sn]
            RT = np.concatenate([extrinsic_params[sn]['r'], extrinsic_params[sn]['t']], axis=-1)
            P = intrinsic_params[sn]['k'] @ RT
            A.append(point[0, 1] * P[2, :] - P[1, :])
            A.append(P[0, :] - point[0, 0] * P[2, :])
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

    return [location, cameras_used]


def locate_sba(cam_sns, camera_coords, intrinsic_params, extrinsic_params):
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

    return [pts_3d, pairs_used]


def create_charuco_boards(plot=False, save_template=False):
    sqWidth = 10  # number of squares width
    sqHeight = 8  # number of squares height

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    board1 = cv2.aruco.CharucoBoard_create(sqWidth, sqHeight, 0.025, 0.0125, dictionary)
    board2 = cv2.aruco.CharucoBoard_create(sqWidth, sqHeight, 0.025, 0.0125, dictionary)
    board2.ids = board2.ids + len(board1.ids)

    imboard1 = board1.draw((991, 792))
    imboard2 = board2.draw((991, 792))
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(2, 1, 1)
        plt.imshow(imboard1, cmap=matplotlib.cm.gray, interpolation="nearest")
        ax.axis("off")
        ax = fig.add_subplot(2, 1, 2)
        plt.imshow(imboard2, cmap=matplotlib.cm.gray, interpolation="nearest")
        ax.axis("off")
        plt.show()
    if save_template:
        cv2.imwrite("../charuco_board_1.png", imboard1)
        cv2.imwrite("../charuco_board_2.png", imboard2)
    return board1, board2


def create_aruco_cube(plot=False, save_template=False):
    markerWidth = 0.045  # This value is in meters
    # Define Aruco board, which can be any 3D shape. See helper CAD file @ https://cad.onshape.com/documents/d51fdec31f121f572b802b11/w/83fac6aaee78bdc978fd804d/e/8ae3ae505e4af3c7402b131a
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)
    board_ids = np.array([[94], [95], [96], [97], [98], [99]], dtype=np.int32)
    board_corners = [
        np.array([[-0.022, 0.023, 0.03], [0.023, 0.022, 0.03], [0.023, -0.023, 0.03], [-0.022, -0.023, 0.03]],
                 dtype=np.float32),
        np.array([[-0.022, -0.03, 0.022], [0.023, -0.03, 0.022], [0.022, -0.03, -0.022], [-0.022, -0.03, -0.022]],
                 dtype=np.float32),
        np.array([[-0.03, -0.023, 0.022], [-0.03, -0.022, -0.023], [-0.03, 0.023, -0.022], [-0.03, 0.023, 0.023]],
                 dtype=np.float32),
        np.array([[-0.022, -0.022, -0.03], [0.023, -0.023, -0.03], [0.023, 0.023, -0.03], [-0.022, 0.023, -0.03]],
                 dtype=np.float32),
        np.array([[0.03, -0.023, -0.022], [0.03, -0.023, 0.023], [0.03, 0.023, 0.022], [0.03, 0.022, -0.023]],
                 dtype=np.float32),
        np.array([[-0.022, 0.03, -0.023], [0.023, 0.03, -0.022], [0.023, 0.03, 0.023], [-0.022, 0.03, 0.022]],
                 dtype=np.float32)
    ]
    board = aruco.Board_create(board_corners, aruco_dict, board_ids)

    imboard = drawArucoBoardPaperTemplate(aruco_dict, markerWidth, board_ids)
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.imshow(imboard, cmap=matplotlib.cm.gray, interpolation="nearest")
        ax.axis("off")
        plt.show()

    if save_template:
        # Create an image from the gridboard
        cv2.imwrite("../aruco_cube.png", imboard)

    return board


def drawArucoBoardPaperTemplate(aruco_dict, markerWidth, board_ids):
    paperPxWidth = 300
    paperWidth = 0.2159  # This value is in meters, for 8.5x11" paper
    faceWidth = 0.06  # This value is in meters
    cubeTemplate = np.ones((int(paperPxWidth * 11 / 8.5), paperPxWidth)) * 255
    markerPx = int((markerWidth / paperWidth) * cubeTemplate.shape[1] / 2)
    facePx = int((faceWidth / paperWidth) * cubeTemplate.shape[1] / 2)

    def drawMarkerAt(id, markerCenter):
        marker = aruco.drawMarker(aruco_dict, id, markerPx * 2, borderBits=1)
        paddedMarker = np.ones((facePx * 2, facePx * 2)) * 255
        padding = facePx - markerPx
        paddedMarker[padding:(facePx * 2) - padding,
        padding:(facePx * 2) - padding] = marker
        cv2.rectangle(paddedMarker, (0, 0), (paddedMarker.shape[0] - 1, paddedMarker.shape[1] - 1), 0, 1)
        cubeTemplate[markerCenter[1] - facePx:markerCenter[1] + facePx,
        markerCenter[0] - facePx:markerCenter[0] + facePx] = paddedMarker

    drawMarkerAt(board_ids[0][0], (int(cubeTemplate.shape[1] / 2), facePx))
    drawMarkerAt(board_ids[1][0], (int(cubeTemplate.shape[1] / 2), facePx + 1 * facePx * 2))
    drawMarkerAt(board_ids[2][0], (int(cubeTemplate.shape[1] / 2) - facePx * 2, facePx + 2 * facePx * 2))
    drawMarkerAt(board_ids[3][0], (int(cubeTemplate.shape[1] / 2), facePx + 2 * facePx * 2))
    drawMarkerAt(board_ids[4][0], (int(cubeTemplate.shape[1] / 2) + facePx * 2, facePx + 2 * facePx * 2))
    drawMarkerAt(board_ids[5][0], (int(cubeTemplate.shape[1] / 2), facePx + 3 * facePx * 2))
    return cubeTemplate


def plot_chessboard_3d(ax, cam_outside_corners, cam_inside_corners, intrinsic_params, extrinsic_params):
    outside_corner_locations = np.zeros((4, 3))
    for idx in range(4):
        img_points = {}
        for sn in cam_outside_corners.keys():
            if len(cam_outside_corners[sn]):
                img_points[sn] = cam_outside_corners[sn][idx, :]
        [outside_corner_locations[idx, :], pairs_used] = locate(list(img_points.keys()), img_points, intrinsic_params,
                                                                extrinsic_params)
    inside_corner_locations = np.zeros((63, 3))
    for idx in range(63):
        img_points = {}
        for sn in cam_inside_corners.keys():
            if len(cam_inside_corners[sn]):
                img_points[sn] = cam_inside_corners[sn][idx, :]
        [inside_corner_locations[idx, :], pairs_used] = locate(list(img_points.keys()), img_points, intrinsic_params,
                                                               extrinsic_params)

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
    return outside_corner_locations


def project_chessboard(cam_board, k, d, rvec, tvec):
    outside_corners = [
        [np.min(cam_board.chessboardCorners[:, 0]), np.min(cam_board.chessboardCorners[:, 1]), 0],
        [np.max(cam_board.chessboardCorners[:, 0]), np.min(cam_board.chessboardCorners[:, 1]), 0],
        [np.min(cam_board.chessboardCorners[:, 0]), np.max(cam_board.chessboardCorners[:, 1]), 0],
        [np.max(cam_board.chessboardCorners[:, 0]), np.max(cam_board.chessboardCorners[:, 1]), 0]]
    outside_corners = np.array(outside_corners)
    [rmat, jac] = cv2.Rodrigues(rvec)
    outside_projected = np.zeros((outside_corners.shape[0], 2))
    for i in range(outside_corners.shape[0]):
        ptProj = np.matmul(rmat, np.reshape(outside_corners[i, :], (3, 1))) + tvec
        [outside_projected[i, :], jac] = cv2.projectPoints(ptProj, (0, 0, 0), (0, 0, 0), k, d)

    inside_corners = cam_board.chessboardCorners
    [rmat, jac] = cv2.Rodrigues(rvec)
    inside_projected = np.zeros((inside_corners.shape[0], 2))
    for i in range(inside_corners.shape[0]):
        ptProj = np.matmul(rmat, np.reshape(inside_corners[i, :], (3, 1))) + tvec
        [inside_projected[i, :], jac] = cv2.projectPoints(ptProj, (0, 0, 0), (0, 0, 0), k, d)

    return outside_projected, inside_projected


if __name__ == '__main__':
    create_charuco_boards(plot=False, save_template=False)
