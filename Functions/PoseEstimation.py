#!/usr/bin/python3
# coding=utf8
import time
import cv2
import mediapipe as mp

from mp_helper import CameraWrapper, FPSMeter


def main():
    camera = CameraWrapper()
    fps_meter = FPSMeter()
    drawing = mp.solutions.drawing_utils
    drawing_styles = mp.solutions.drawing_styles
    pose = mp.solutions.pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    cv2.namedWindow('MediaPipe Pose', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('MediaPipe Pose', 960, 720)

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_image)
            if results.pose_landmarks:
                drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS,
                    drawing_styles.get_default_pose_landmarks_style()
                )

            fps_meter.update()
            fps_meter.draw(frame)
            cv2.putText(frame, "ESC to exit", (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow('MediaPipe Pose', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
    finally:
        camera.release()
        pose.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
