#!/usr/bin/python3
# coding=utf8
import time
import cv2
import hiwonder.Camera as Camera
import hiwonder.yaml_handle as yaml_handle


class FPSMeter:
    def __init__(self, average_interval=1.0):
        self.interval = max(average_interval, 0.1)
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0

    def update(self):
        self.frame_count += 1
        now = time.time()
        if now - self.start_time >= self.interval:
            self.fps = self.frame_count / (now - self.start_time)
            self.frame_count = 0
            self.start_time = now

    def draw(self, frame, origin=(10, 26), color=(0, 255, 0)):
        cv2.putText(frame, "FPS: {:.2f}".format(self.fps), origin,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame


class CameraWrapper:
    def __init__(self, resolution=(640, 480)):
        self.threaded = False
        self.camera = None
        config = yaml_handle.get_yaml_data('/boot/camera_setting.yaml')
        self.flip = bool(config.get('flip', False))
        self.flip_param = int(config.get('flip_param', 0))
        if config.get('open_once'):
            self.camera = cv2.VideoCapture('http://127.0.0.1:8080/?action=stream?dummy=param.mjpg')
        else:
            self.threaded = True
            self.camera = Camera.Camera(resolution)
            self.camera.camera_open()

    def read(self):
        if self.threaded:
            ret, frame = self.camera.read()
        else:
            ret, frame = self.camera.read()
        if ret and frame is not None and self.flip:
            frame = cv2.flip(frame, self.flip_param)
        return ret, frame

    def release(self):
        if self.threaded:
            self.camera.camera_close()
        else:
            if self.camera is not None:
                self.camera.release()

