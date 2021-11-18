from ximea import xiapi
import cv2
import sys

try:
    fr_disp = int(sys.argv[1])
except:
    fr_disp = 30

try:
    fr_shutter = int(sys.argv[2])
except:
    fr_shutter = 200


try:
    set_gain = int(sys.argv[3])
except:
    set_gain = 5

def quick_resize(data, scale, og_width, og_height):
    width = int(og_width * scale)
    height = int(og_height * scale)
    dim = (width, height)
    resized = cv2.resize(
        data,
        dim,
        interpolation=cv2.INTER_AREA
    )
    return resized


actual_framerate = fr_disp
shutter_framerate = fr_shutter
shutter = int((1/shutter_framerate)*1e+6)-100
gain = set_gain
f_size = (1280, 1024)

cam0 = xiapi.Camera()
cam0.disable_auto_bandwidth_calculation()
cam0.open_device_by_SN("06955451")
cam0.set_sensor_feature_value(1)
cam0.set_exposure(shutter)
cam0.enable_auto_wb()
cam0.set_gain(gain)
cam0.set_acq_timing_mode("XI_ACQ_TIMING_MODE_FRAME_RATE")
cam0.set_framerate(actual_framerate)
cam0.set_imgdataformat("XI_RGB24")
cam0.set_output_bit_depth("XI_BPP_8")

cam1 = xiapi.Camera()
cam1.disable_auto_bandwidth_calculation()
cam1.open_device_by_SN("32052251")
cam1.set_sensor_feature_value(1)
cam1.set_exposure(shutter)
cam1.enable_auto_wb()
cam1.set_gain(gain)
cam1.set_acq_timing_mode("XI_ACQ_TIMING_MODE_FRAME_RATE")
cam1.set_framerate(actual_framerate)
cam1.set_imgdataformat("XI_RGB24")
cam1.set_output_bit_depth("XI_BPP_8")

cam2 = xiapi.Camera()
cam2.disable_auto_bandwidth_calculation()
cam2.open_device_by_SN("39050251")
cam2.set_sensor_feature_value(1)
cam2.set_exposure(shutter)
cam2.enable_auto_wb()
cam2.set_gain(gain)
cam2.set_acq_timing_mode("XI_ACQ_TIMING_MODE_FRAME_RATE")
cam2.set_framerate(actual_framerate)
cam2.set_imgdataformat("XI_RGB24")
cam2.set_output_bit_depth("XI_BPP_8")

cam3 = xiapi.Camera()
cam3.disable_auto_bandwidth_calculation()
cam3.open_device_by_SN("32050651")
cam3.set_sensor_feature_value(1)
cam3.set_exposure(shutter)
cam3.enable_auto_wb()
cam3.set_gain(gain)
cam3.set_acq_timing_mode("XI_ACQ_TIMING_MODE_FRAME_RATE")
cam3.set_framerate(actual_framerate)
cam3.set_imgdataformat("XI_RGB24")
cam3.set_output_bit_depth("XI_BPP_8")

img0 = xiapi.Image()
img1 = xiapi.Image()
img2 = xiapi.Image()
img3 = xiapi.Image()

font = cv2.FONT_HERSHEY_COMPLEX_SMALL

cam0.start_acquisition()
cam1.start_acquisition()
cam2.start_acquisition()
cam3.start_acquisition()

try:
    while True:
        cam0.get_image(img0)
        ci0 = quick_resize(img0.get_image_data_numpy(), 0.4, f_size[0], f_size[1])
        cv2.putText(ci0, "cam 0", (10,20), font, 1, (0, 0, 0), 1)
        
        cam1.get_image(img1)
        ci1 = quick_resize(img1.get_image_data_numpy(), 0.4, f_size[0], f_size[1])
        cv2.putText(ci1, "cam 1", (10,20), font, 1, (0, 0, 0), 1)
        
        cam2.get_image(img2)
        ci2 = quick_resize(img2.get_image_data_numpy(), 0.4, f_size[0], f_size[1])
        cv2.putText(ci2, "cam 2", (10,20), font, 1, (0, 0, 0), 1)

        cam3.get_image(img3)
        ci3 = quick_resize(img3.get_image_data_numpy(), 0.4, f_size[0], f_size[1])
        cv2.putText(ci3, "cam 3", (10,20), font, 1, (0, 0, 0), 1)

        final_frame = cv2.vconcat(
            [cv2.hconcat([ci0, ci1]), cv2.hconcat([ci2, ci3])]
        )

        cv2.imshow("cam", final_frame)
        cv2.waitKey(1)
except KeyboardInterrupt:
    cv2.destroyAllWindows()

cam0.stop_acquisition()
cam0.close_device()
cam1.stop_acquisition()
cam1.close_device()
cam2.stop_acquisition()
cam2.close_device()
cam3.stop_acquisition()
cam3.close_device()


        
