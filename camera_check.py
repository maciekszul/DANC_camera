import json

from camera_io import init_camera_sources, shtr_spd
from utilities.tools import quick_resize
import cv2
import sys

try:
    json_file = sys.argv[1]
except:
    json_file = "settings.json"
    
try:
    fr_disp = int(sys.argv[2])
except:
    fr_disp = 30

try:
    fr_shutter = int(sys.argv[3])
except:
    fr_shutter = 200

try:
    set_gain = int(sys.argv[4])
except:
    set_gain = 5


# opening a json file
with open(json_file) as settings_file:
    params = json.load(settings_file)

actual_framerate = fr_disp
shutter = shtr_spd(actual_framerate)
gain = set_gain
f_size = (1280, 1024)

cams = init_camera_sources(params, actual_framerate, shutter, gain, sensor_feature_value=1, disable_auto_bandwidth=True,
                           img_data_format="XI_RGB24", output_bit_depth='XI_BPP_8', auto_wb=True)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

try:
    while True:
        cam_datas=[]
        for cam in cams:
            cam_data=cam.next_frame()
            ci0 = quick_resize(cam_data, 0.4, f_size[0], f_size[1])
            cam_data = cv2.putText(cam_data, "cam %s" % cam.sn, (10, 20), font, 1, (0, 0, 0), 1)
            cam_datas.append(cam_data)

        final_frame = cv2.vconcat(
            [cv2.hconcat([cam_datas[0], cam_datas[1]]), cv2.hconcat([cam_datas[2], cam_datas[3]])]
        )

        cv2.imshow("cam", final_frame)
        cv2.waitKey(1)
except KeyboardInterrupt:
    cv2.destroyAllWindows()

for cam in cams:
    cam.close()
