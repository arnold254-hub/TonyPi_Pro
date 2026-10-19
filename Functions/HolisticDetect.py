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
    holistic = mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    cv2.namedWindow('MediaPipe Holistic', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('MediaPipe Holistic', 960, 720)

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb_image)

            if results.face_landmarks:
                drawing.draw_landmarks(
                    frame,
                    results.face_landmarks,
                    mp.solutions.holistic.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style()
                )
            if results.left_hand_landmarks:
                drawing.draw_landmarks(
                    frame,
                    results.left_hand_landmarks,
                    mp.solutions.holistic.HAND_CONNECTIONS,
                    drawing_styles.get_default_hand_landmarks_style(),
                    drawing_styles.get_default_hand_connections_style()
                )
            if results.right_hand_landmarks:
                drawing.draw_landmarks(
                    frame,
                    results.right_hand_landmarks,
                    mp.solutions.holistic.HAND_CONNECTIONS,
                    drawing_styles.get_default_hand_landmarks_style(),
                    drawing_styles.get_default_hand_connections_style()
                )
            if results.pose_landmarks:
                drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp.solutions.holistic.POSE_CONNECTIONS,
                    drawing_styles.get_default_pose_landmarks_style()
                )

            fps_meter.update()
            fps_meter.draw(frame)
            cv2.putText(frame, "ESC to exit", (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow('MediaPipe Holistic', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
    finally:
        camera.release()
        holistic.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
