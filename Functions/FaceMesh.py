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
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    cv2.namedWindow('MediaPipe Face Mesh', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('MediaPipe Face Mesh', 960, 720)

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_image)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style()
                    )
                    drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style()
                    )

            fps_meter.update()
            fps_meter.draw(frame)
            cv2.putText(frame, "ESC to exit", (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow('MediaPipe Face Mesh', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
    finally:
        camera.release()
        face_mesh.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
