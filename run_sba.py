import json
import pickle
import sys

import numpy as np

from utilities.calib_tools import bundle_adjust_points_and_extrinsics

try:
    json_file = sys.argv[1]
    print("USING: ", json_file)
except:
    json_file = "settings.json"
    print("USING: ", json_file)

# opening a json file
with open(json_file) as settings_file:
    params = json.load(settings_file)

# Open intrinsic parameters
handle = open('intrinsic_params.pickle', "rb")
intrinsic_params = pickle.load(handle)
handle.close()

# Open extrinsic parameters
handle = open('extrinsic_params.pickle', "rb")
extrinsic_params = pickle.load(handle)
handle.close()

# Open SBA data
handle = open('sba_data.pickle', "rb")
sba_data = pickle.load(handle)
handle.close()

# Convert parameters from dictionaries to arrays
k_arr = np.array([intrinsic_params[x]['k'] for x in params['cam_sns']])
d_arr = np.array([intrinsic_params[x]['d'] for x in params['cam_sns']])
r_arr = np.array([extrinsic_params[x]['r'] for x in params['cam_sns']])
t_arr = np.array([extrinsic_params[x]['t'] for x in params['cam_sns']])

# Run SBA
obj_pts, r_arr, t_arr, res = bundle_adjust_points_and_extrinsics(np.squeeze(sba_data['points_2d']),
                                                                 np.squeeze(sba_data['points_3d']),
                                                                 sba_data['points_3d_indices'],
                                                                 sba_data['camera_indices'], k_arr, d_arr, r_arr, t_arr)
print(f"\nBefore: mean: {np.mean(res['before'])}, std: {np.std(res['before'])}")
print(f"After: mean: {np.mean(res['after'])}, std: {np.std(res['after'])}\n")

# Save new extrinsic parameters
extrinsic_sba_params = {}
for cam_idx, sn in enumerate(params['cam_sns']):
    extrinsic_sba_params[sn] = {
        'r': np.squeeze(r_arr[cam_idx, :, :]),
        't': t_arr[cam_idx, :, :]
    }
pickle.dump(
    extrinsic_sba_params,
    open(
        "extrinsic_sba_params.pickle",
        "wb",
    ),
)
