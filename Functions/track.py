#!/usr/bin/python3
# coding=utf8
import os
import cv2
import time
import torch
import numpy as np
import onnxruntime as ort

import hiwonder.PID as PID
import hiwonder.Camera as Camera
import hiwonder.yaml_handle as yaml_handle
import hiwonder.ros_robot_controller_sdk as rrc
from hiwonder.Controller import Controller

from collections import namedtuple


class Config:
    TRACK_CONTEXT_AMOUNT = 0.5
    TRACK_EXEMPLAR_SIZE = 127
    TRACK_INSTANCE_SIZE = 255
    TRACK_OUTPUT_SIZE = 15

    TRACK_PENALTY_K = 0.16
    TRACK_WINDOW_INFLUENCE = 0.46
    TRACK_LR = 0.34

    POINT_STRIDE = 8


config = Config
Corner = namedtuple('Corner', 'x1 y1 x2 y2')
Center = namedtuple('Center', 'x y w h')


def corner2center(corner):
    if isinstance(corner, Corner):
        x1, y1, x2, y2 = corner
        return Center((x1 + x2) * 0.5, (y1 + y2) * 0.5, (x2 - x1), (y2 - y1))
    else:
        x1, y1, x2, y2 = corner[0], corner[1], corner[2], corner[3]
        x = (x1 + x2) * 0.5
        y = (y1 + y2) * 0.5
        w = x2 - x1
        h = y2 - y1
        return x, y, w, h


class BackBoneProcessor(object):
    def __init__(self, input_dtype):
        if isinstance(input_dtype, list):
            self.in_dtype = input_dtype[0]
        else:
            self.in_dtype = input_dtype

    def pre(self, input):
        return input

    def post(self, output):
        if len(output) != 1:
            raise ValueError("engine error!")

        return output[0]


class HeadProcessor(object):
    def __init__(self, input_dtype):
        if isinstance(input_dtype, list):
            self.in_dtype = input_dtype[0]
        else:
            self.in_dtype = input_dtype

    def pre(self, input1, input2):
        return [input1, input2]

    def post(self, output1, output2):
        output1 = torch.from_numpy(output1)
        output2 = torch.from_numpy(output2)

        output1 = torch.reshape(output1, [1, 2, 15, 15])
        output2 = torch.reshape(output2, [1, 4, 15, 15])

        return output1, output2


class ONNXEngine(object):
    def __init__(self, model_path=""):
        if not os.path.exists(model_path):
            raise FileNotFoundError('ONNX model file {} is not found'.format(model_path))

        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]

        self.input_dtypes = []
        for input_info in self.session.get_inputs():
            if input_info.type == 'tensor(float)':
                self.input_dtypes.append(np.float32)
            elif input_info.type == 'tensor(double)':
                self.input_dtypes.append(np.float64)
            else:
                self.input_dtypes.append(np.float32)

    def __call__(self, data):
        if isinstance(data, list):
            if len(data) != len(self.input_names):
                raise ValueError("Input data length does not match model inputs")

            input_dict = {}
            for i, input_name in enumerate(self.input_names):
                input_dict[input_name] = data[i].astype(self.input_dtypes[i])

            outputs = self.session.run(self.output_names, input_dict)
            return outputs
        else:
            input_dict = {self.input_names[0]: data.astype(self.input_dtypes[0])}
            outputs = self.session.run(self.output_names, input_dict)
            return outputs

    def get_input_dtype(self):
        return self.input_dtypes


class Model(object):
    def __init__(self, exam_back_engine_path="", temp_back_engine_path="", head_engine_path=""):
        self.back_exam_engine = ONNXEngine(exam_back_engine_path)
        self.back_temp_engine = ONNXEngine(temp_back_engine_path)
        self.head_engine = ONNXEngine(head_engine_path)

        self.back_exam_processor = BackBoneProcessor(self.back_exam_engine.get_input_dtype())
        self.back_temp_processor = BackBoneProcessor(self.back_temp_engine.get_input_dtype())
        self.head_processor = HeadProcessor(self.head_engine.get_input_dtype())

    def template(self, z):
        zf = self.back_temp_engine(self.back_temp_processor.pre(z))
        self.zf = self.back_temp_processor.post(zf)

    def track(self, x):
        xf = self.back_exam_engine(self.back_exam_processor.pre(x))

        cls, loc = self.head_engine(self.head_processor.pre(self.zf, self.back_exam_processor.post(xf)))

        cls, loc = self.head_processor.post(cls, loc)

        return {
            'cls': cls,
            'loc': loc,
        }


class BaseTracker(object):
    def init(self, img, bbox):
        raise NotImplementedError

    def track(self, img):
        raise NotImplementedError


class SiameseTracker(BaseTracker):
    def get_subwindow(self, im, pos, model_sz, original_sz, avg_chans):
        if isinstance(pos, float):
            pos = [pos, pos]

        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                             int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                          int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.float32)
        return im_patch


class Tracker(SiameseTracker):
    def __init__(self, back_exam_engine_path="", temp_exam_engine_path="", head_engine_path=""):
        self.score_size = config.TRACK_OUTPUT_SIZE

        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.cls_out_channels = 2
        self.window = window.flatten()

        self.points = self.generate_points(config.POINT_STRIDE, self.score_size)
        self.model = Model(back_exam_engine_path, temp_exam_engine_path, head_engine_path)

    def generate_points(self, stride, size):
        ori = -(size // 2) * stride
        x, y = np.meshgrid([ori + stride * dx for dx in np.arange(0, size)],
                           [ori + stride * dy for dy in np.arange(0, size)])
        points = np.zeros((size * size, 2), dtype=np.float32)
        points[:, 0], points[:, 1] = x.astype(np.float32).flatten(), y.astype(np.float32).flatten()

        return points

    def _convert_bbox(self, delta, point):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.detach().cpu().numpy()

        delta[0, :] = point[:, 0] - delta[0, :]
        delta[1, :] = point[:, 1] - delta[1, :]
        delta[2, :] = point[:, 0] + delta[2, :]
        delta[3, :] = point[:, 1] + delta[3, :]
        delta[0, :], delta[1, :], delta[2, :], delta[3, :] = corner2center(delta)
        return delta

    def _convert_score(self, score):
        if self.cls_out_channels == 1:
            score = score.permute(1, 2, 3, 0).contiguous().view(-1)
            score = score.sigmoid().detach().cpu().numpy()
        else:
            score = score.permute(1, 2, 3, 0).contiguous().view(self.cls_out_channels, -1).permute(1, 0)
            score = score.softmax(1).detach()[:, 1].cpu().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox):
        self.center_pos = np.array([bbox[0] + (bbox[2] - 1) / 2,
                                    bbox[1] + (bbox[3] - 1) / 2])

        self.size = np.array([bbox[2], bbox[3]])

        w_z = self.size[0] + config.TRACK_CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + config.TRACK_CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        self.channel_average = np.mean(img, axis=(0, 1))

        z_crop = self.get_subwindow(img, self.center_pos,
                                    config.TRACK_EXEMPLAR_SIZE,
                                    s_z, self.channel_average)

        self.model.template(z_crop)

    def track(self, img):
        w_z = self.size[0] + config.TRACK_CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + config.TRACK_CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))
        scale_z = config.TRACK_EXEMPLAR_SIZE / s_z
        s_x = s_z * (config.TRACK_INSTANCE_SIZE / config.TRACK_EXEMPLAR_SIZE)

        x_crop = self.get_subwindow(img, self.center_pos,
                                    config.TRACK_INSTANCE_SIZE,
                                    round(s_x), self.channel_average)

        outputs = self.model.track(x_crop)

        score = self._convert_score(outputs['cls'])
        bbox = self._convert_bbox(outputs['loc'], self.points)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        s_c = change(sz(bbox[2, :], bbox[3, :]) /
                     (sz(self.size[0] * scale_z, self.size[1] * scale_z)))

        r_c = change((self.size[0] / self.size[1]) /
                     (bbox[2, :] / bbox[3, :]))

        penalty = np.exp(-(r_c * s_c - 1) * config.TRACK_PENALTY_K)

        pscore = penalty * score

        pscore = pscore * (1 - config.TRACK_WINDOW_INFLUENCE) + \
            self.window * config.TRACK_WINDOW_INFLUENCE

        best_idx = np.argmax(pscore)

        bbox = bbox[:, best_idx] / scale_z

        lr = penalty[best_idx] * score[best_idx] * config.TRACK_LR
        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        cx, cy, width, height = self._bbox_clip(cx, cy, width, height, img.shape[:2])

        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]

        best_score = score[best_idx]

        return best_score, list(map(int, bbox))


class MouseSelector:
    def __init__(self, window_name):
        self.window_name = window_name
        self.dragging = False
        self.start_point = None
        self.end_point = None
        self.ready_bbox = None
        cv2.setMouseCallback(window_name, self._callback)

    def _callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.start_point = (x, y)
            self.end_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.end_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.end_point = (x, y)
            if self.start_point:
                x0 = min(self.start_point[0], self.end_point[0])
                y0 = min(self.start_point[1], self.end_point[1])
                x1 = max(self.start_point[0], self.end_point[0])
                y1 = max(self.start_point[1], self.end_point[1])
                if (x1 - x0) > 8 and (y1 - y0) > 8:
                    self.ready_bbox = (x0, y0, x1 - x0, y1 - y0)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.reset()

    def reset(self):
        self.dragging = False
        self.start_point = None
        self.end_point = None
        self.ready_bbox = None

    def poll_selection(self):
        if self.ready_bbox is None:
            return None
        bbox = self.ready_bbox
        self.ready_bbox = None
        return bbox

    def draw(self, image):
        if self.dragging and self.start_point and self.end_point:
            cv2.rectangle(image, self.start_point, self.end_point, (0, 255, 255), 2)
        return image


class NanotrackFollower:
    def __init__(self):
        self.window_name = 'track'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 960, 720)
        self.selector = MouseSelector(self.window_name)

        servo_data = yaml_handle.get_yaml_data(yaml_handle.servo_file_path)
        self.pitch_center = 1500
        self.yaw_center = servo_data.get('servo2', 1500)
        self.pitch = self.pitch_center
        self.yaw = self.yaw_center
        self.pitch_limits = (1000, 2000)
        self.yaw_limits = (500, 2500)

        self.board = rrc.Board()
        self.controller = Controller(self.board)
        self.controller.set_pwm_servo_pulse(1, 1500, 500)
        self.controller.set_pwm_servo_pulse(2, self.yaw, 500)

        self.x_pid = PID.PID(P=0.2, I=0.0, D=0.0007)
        self.y_pid = PID.PID(P=0.2, I=0.0, D=0.0007)

        script_dir = os.path.split(os.path.realpath(__file__))[0]
        model_dir = os.path.join(script_dir, 'model', 'nanotrack')
        back_exam_path = os.path.join(model_dir, 'nanotrack_backbone_exam.onnx')
        back_temp_path = os.path.join(model_dir, 'nanotrack_backbone_temp.onnx')
        head_path = os.path.join(model_dir, 'nanotrack_head.onnx')
        self.tracker = Tracker(back_exam_path, back_temp_path, head_path)
        self.tracking = False
        self.last_update = time.time()
        self.update_interval = 0.05
        self.score_threshold = 0.45
        self.current_bbox = None

        camera_setting = yaml_handle.get_yaml_data('/boot/camera_setting.yaml')
        if camera_setting.get('open_once'):
            self.camera = cv2.VideoCapture('http://127.0.0.1:8080/?action=stream?dummy=param.mjpg')
            self.thread_camera = False
        else:
            self.camera = Camera.Camera()
            self.camera.camera_open()
            self.thread_camera = True

    def close(self):
        if self.thread_camera:
            self.camera.camera_close()
        else:
            self.camera.release()
        cv2.destroyAllWindows()

    def read_frame(self):
        if self.thread_camera:
            ret, frame = self.camera.read()
            return ret, frame
        return self.camera.read()

    def reset_tracking(self):
        self.tracking = False
        self.current_bbox = None
        self.x_pid.clear()
        self.y_pid.clear()

    def clamp(self, value, limits):
        return min(max(value, limits[0]), limits[1])

    def move_servos(self, cx, cy, width, height):
        now = time.time()
        if now - self.last_update < self.update_interval:
            return
        self.last_update = now

        x_center = width / 2.0
        y_center = height / 2.0

        if abs(cx - x_center) < 2 and abs(cy - y_center) < 2:
            return

        self.x_pid.SetPoint = x_center
        self.y_pid.SetPoint = y_center

        self.x_pid.update(cx)
        self.y_pid.update(cy)

        dx = int(self.x_pid.output)
        dy = int(self.y_pid.output)

        if dx == 0 and dy == 0:
            return

        move_time = max(abs(dx), abs(dy)) * 0.00025
        move_time = float(max(0.02, min(move_time, 0.2)))

        self.yaw += dx
        self.pitch += dy

        self.yaw = self.clamp(self.yaw, self.yaw_limits)
        self.pitch = self.clamp(self.pitch, self.pitch_limits)

        duration = int(move_time * 1000)
        self.controller.set_pwm_servo_pulse(2, self.yaw, duration)
        self.controller.set_pwm_servo_pulse(1, self.pitch, duration)

    def draw_info(self, frame, tracking_ok, score):
        info = "Drag to select target, SPACE to clear, ESC to exit"
        cv2.putText(frame, info, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        status_text = "Tracking Score:{:.2f}".format(score) if tracking_ok else "Awaiting target"
        cv2.putText(frame, status_text, (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if tracking_ok else (0, 0, 255), 2)
        return frame

    def run(self):
        try:
            while True:
                ret, frame = self.read_frame()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                display = frame.copy()
                new_bbox = self.selector.poll_selection()
                if new_bbox is not None:
                    self.tracker.init(frame, new_bbox)
                    self.tracking = True
                    self.current_bbox = new_bbox
                    self.x_pid.clear()
                    self.y_pid.clear()

                tracking_ok = False
                score = 0.0

                if self.tracking:
                    score, bbox = self.tracker.track(frame)
                    if score >= self.score_threshold:
                        x, y, w, h = bbox
                        x = max(0, x)
                        y = max(0, y)
                        w = max(1, w)
                        h = max(1, h)
                        if (x + w) > frame.shape[1]:
                            w = frame.shape[1] - x
                        if (y + h) > frame.shape[0]:
                            h = frame.shape[0] - y
                        self.current_bbox = [x, y, w, h]
                        p1 = (int(x), int(y))
                        p2 = (int(x + w), int(y + h))
                        cv2.rectangle(display, p1, p2, (0, 255, 0), 2)
                        cx = x + w / 2.0
                        cy = y + h / 2.0
                        cv2.circle(display, (int(cx), int(cy)), 5, (0, 255, 255), -1)
                        tracking_ok = True
                        self.move_servos(cx, cy, frame.shape[1], frame.shape[0])
                    else:
                        cv2.putText(display, "Target lost, please reselect", (10, frame.shape[0] - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                display = self.selector.draw(display)
                display = self.draw_info(display, tracking_ok, score)

                cv2.imshow(self.window_name, display)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                if key == 32:
                    self.selector.reset()
                    self.reset_tracking()

        finally:
            self.close()


def main():
    follower = NanotrackFollower()
    follower.run()


if __name__ == '__main__':
    main()
