import os.path
import time

import cv2
from ximea import xiapi


class FrameSource:

    def __init__(self):
        pass

    def init(self):
        pass

    def next_frame(self):
        return None

    def close(self):
        pass


class CameraSource(FrameSource):
    def __init__(self, sn, fps, shutter, gain):
        FrameSource.__init__(self)
        self.sn = sn
        self.fps = fps
        self.shutter = shutter
        self.gain = gain
        self.cam = None
        self.img = None

    def init(self, sensor_feature_value=None, disable_auto_bandwidth=False, img_data_format="XI_RGB32",
             output_bit_depth=None, auto_wb=True, counter_selector=None):
        try:
            self.cam = xiapi.Camera()
            self.cam.open_device_by_SN(self.sn)  # put a serial number of the camera
            if sensor_feature_value is not None:
                self.cam.set_sensor_feature_value(sensor_feature_value)
            self.cam.set_imgdataformat(img_data_format)
            if disable_auto_bandwidth:
                self.cam.disable_auto_bandwidth_calculation()
            if auto_wb:
                self.cam.enable_auto_wb()
            else:
                self.cam.disable_auto_wb()
            if counter_selector is not None:
                self.cam.set_counter_selector(counter_selector)
            if output_bit_depth is not None:
                self.cam.set_output_bit_depth(output_bit_depth)
            self.cam.set_acq_timing_mode("XI_ACQ_TIMING_MODE_FRAME_RATE")
            self.cam.set_framerate(self.fps)
            self.cam.set_exposure(self.shutter)
            self.cam.set_gain(self.gain)
            self.img = xiapi.Image()
            self.cam.start_acquisition()

        except:
            self.cam = None
            self.img = None

    def next_frame(self):
        self.cam.get_image(self.img)
        cam_data = self.img.get_image_data_numpy()
        return cam_data

    def close(self):
        self.cam.stop_acquisition()
        self.cam.close_device()


class FileSource(FrameSource):
    def __init__(self, sn, filename):
        FrameSource.__init__(self)
        self.sn = sn
        self.filename = filename
        self.vidcap = None

    def init(self):
        if os.path.exists(self.filename):
            self.vidcap = cv2.VideoCapture(self.filename)
        else:
            self.vidcap = None

    def next_frame(self):
        success, image = self.vidcap.read()
        if success:
            return image
        return None


def init_camera_sources(parameters, fps, shutter, gain, sensor_feature_value=None, disable_auto_bandwidth=False,
                        img_data_format="XI_RGB32", output_bit_depth=None, auto_wb=True, counter_selector=None):
    srcs = []
    ############################
    # for each camera separately
    for sn in parameters['cam_sns']:
        src = CameraSource(sn, fps, shutter, gain)
        src.init(sensor_feature_value=sensor_feature_value, disable_auto_bandwidth=disable_auto_bandwidth,
                 img_data_format=img_data_format, output_bit_depth=output_bit_depth, auto_wb=auto_wb,
                 counter_selector=counter_selector)
        if src.cam is not None:
            srcs.append(src)
    return srcs


def init_file_sources(parameters, prefix):
    srcs = []
    for sn in parameters['cam_sns']:
        src = FileSource(sn, '%s_cam%s.avi' % (prefix, sn))
        src.init()
        if src.vidcap is not None:
            srcs.append(src)
    return srcs


def shtr_spd(framerate):
    return int((1 / framerate) * 1e+6) - 100


def get_wb_coef(s_n, framerate, shutter, gain):
    cam = xiapi.Camera()
    img = xiapi.Image()
    cam.open_device_by_SN(s_n)
    cam.set_sensor_feature_value(1)
    cam.set_imgdataformat("XI_RGB24")
    cam.set_acq_timing_mode("XI_ACQ_TIMING_MODE_FRAME_RATE")
    cam.set_framerate(framerate)
    cam.set_exposure(shutter)
    cam.set_gain(gain)
    cam.enable_auto_wb()
    cam.start_acquisition()
    start = time.monotonic()
    while (time.monotonic() - start) <= 1:
        cam.get_image(img)

    kr = cam.get_wb_kr()
    kg = cam.get_wb_kg()
    kb = cam.get_wb_kb()

    cam.stop_acquisition()
    cam.close_device()

    return kr, kg, kb
