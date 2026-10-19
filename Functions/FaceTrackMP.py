#!/usr/bin/python3
# coding=utf8
import time
import cv2
import mediapipe as mp

import hiwonder.PID as PID
import hiwonder.yaml_handle as yaml_handle
import hiwonder.ros_robot_controller_sdk as rrc
from hiwonder.Controller import Controller

from mp_helper import CameraWrapper, FPSMeter


class ServoFaceTracker:
    def __init__(self):
        servo_data = yaml_handle.get_yaml_data(yaml_handle.servo_file_path)
        self.pitch_center = 1500
        self.yaw_center = servo_data.get('servo2', 1500)
        self.pitch = self.pitch_center
        self.yaw = self.yaw_center
        self.pitch_limits = (500, 2500)
        self.yaw_limits = (500, 2500)

        self.board = rrc.Board()
        self.controller = Controller(self.board)
        self.controller.set_pwm_servo_pulse(1, 1500, 500)
        self.controller.set_pwm_servo_pulse(2, self.yaw, 500)

        time.sleep(1)

        self.pid_x = PID.PID(P=0.14, I=0.0, D=0.0007)
        self.pid_y = PID.PID(P=0.14, I=0.0, D=0.0007)

        self.update_interval = 0.05
        self.last_update = time.time()

    @staticmethod
    def clamp(value, limits):
        return max(limits[0], min(limits[1], value))

    def move_to_center(self, duration=400):
        self.pitch = self.clamp(self.pitch_center, self.pitch_limits)
        self.yaw = self.clamp(self.yaw_center, self.yaw_limits)
        self.pid_x.clear()
        self.pid_y.clear()
        self.controller.set_pwm_servo_pulse(1, 1500, duration)
        self.controller.set_pwm_servo_pulse(2, self.yaw, duration)

    def update(self, target_x, target_y, width, height):
        now = time.time()
        if now - self.last_update < self.update_interval:
            return
        self.last_update = now

        center_x = width / 2.0
        center_y = height / 2.0
        error_x = target_x - center_x
        error_y = target_y - center_y

        self.pid_x.SetPoint = center_x
        self.pid_y.SetPoint = center_y
        self.pid_x.update(target_x)
        self.pid_y.update(target_y)

        delta_yaw = int(self.pid_x.output)
        delta_pitch = -int(self.pid_y.output)

        if abs(error_x) < 20:
            delta_yaw = 0
        if abs(error_y) < 35:
            delta_pitch = 0

        delta_yaw = max(-20, min(20, delta_yaw))
        delta_pitch = max(-25, min(25, delta_pitch))

        applied_pitch_delta = -delta_pitch

        if delta_yaw == 0 and applied_pitch_delta == 0:
            print("目标中心({:.1f},{:.1f}) 误差(dx={:.1f}, dy={:.1f}) 控制量(delta_yaw=0, delta_pitch=0) 保持当前位置(yaw={}, pitch={})".format(
                target_x, target_y, error_x, error_y, self.yaw, self.pitch))
            return

        move_time = float(max(abs(delta_yaw), abs(applied_pitch_delta)) * 0.00025)
        move_time = max(0.02, min(move_time, 0.2))
        duration = int(move_time * 1000)

        self.yaw = self.clamp(self.yaw + delta_yaw, self.yaw_limits)
        self.pitch = self.clamp(self.pitch + applied_pitch_delta, self.pitch_limits)
        self.controller.set_pwm_servo_pulse(2, self.yaw, duration)
        self.controller.set_pwm_servo_pulse(1, self.pitch, duration)
        print("目标中心({:.1f},{:.1f}) 误差(dx={:.1f}, dy={:.1f}) 控制量(delta_yaw={}, delta_pitch={}) 舵机输出(yaw={}, pitch={}, duration={}ms)".format(
            target_x, target_y, error_x, error_y, delta_yaw, applied_pitch_delta, self.yaw, self.pitch, duration))


def select_best_detection(detections):
    best = None
    best_score = -1.0
    for detection in detections:
        score = detection.score[0] if detection.score else 0.0
        if score > best_score:
            best = detection
            best_score = score
    return best, best_score


def detection_to_bbox(detection, frame_shape):
    h, w = frame_shape[:2]
    bbox = detection.location_data.relative_bounding_box
    x = max(int(bbox.xmin * w), 0)
    y = max(int(bbox.ymin * h), 0)
    width = max(int(bbox.width * w), 1)
    height = max(int(bbox.height * h), 1)
    x = min(x, w - 1)
    y = min(y, h - 1)
    width = min(width, w - x)
    height = min(height, h - y)
    return x, y, width, height


def main():
    camera = CameraWrapper()
    fps_meter = FPSMeter()
    tracker = ServoFaceTracker()
    face_detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
    cv2.namedWindow('MediaPipe Face Tracking', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('MediaPipe Face Tracking', 960, 720)

    lost_frames = 0
    LOST_LIMIT = 20

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detector.process(rgb_image)
            tracking_ok = False
            score = 0.0

            if results.detections:
                detection, score = select_best_detection(results.detections)
                if detection and score >= 0.7:
                    x, y, w, h = detection_to_bbox(detection, frame.shape)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cx = x + w / 2.0
                    cy = y + h / 2.0
                    cv2.circle(frame, (int(cx), int(cy)), 5, (0, 255, 255), -1)
                    tracker.update(cx, cy, frame.shape[1], frame.shape[0])
                    lost_frames = 0
                    tracking_ok = True
                else:
                    lost_frames += 1
            else:
                lost_frames += 1

            if lost_frames > LOST_LIMIT:
                lost_frames = 0
                tracker.move_to_center()

            fps_meter.update()
            fps_meter.draw(frame)
            status_text = "Tracking Score:{:.2f}".format(score) if tracking_ok else "Searching..."
            cv2.putText(frame, status_text, (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if tracking_ok else (0, 0, 255), 2)
            cv2.putText(frame, "ESC to exit", (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow('MediaPipe Face Tracking', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
    finally:
        camera.release()
        face_detector.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
