# app_multi.py
import os
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)

import time
import cv2
from person_detector import PersonDetector
from multiperson_posture import MultiPersonPosture
from phone_detector import PhoneDetector  # add this import


REQ_W, REQ_H, REQ_FPS = 1280, 720, 60
SHOW_FPS = True

def open_cam(index=0, width=1280, height=720, fps=60):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {index}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS,          fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    got_f = cap.get(cv2.CAP_PROP_FPS)
    print(f"[Camera] Requested: {width}x{height}@{fps} | Got: {got_w}x{got_h}@{got_f:.1f}")
    return cap

def main():
    people_det = PersonDetector(weights="yolov8n.pt", conf=0.35, imgsz=480, max_det=6)
    mp_multi = MultiPersonPosture(
        max_faces_per_person=1,
        max_hands_per_person=2,
        face_detect_conf=0.5, face_track_conf=0.5,
        hand_detect_conf=0.35, hand_track_conf=0.5,
        pose_detect_conf=0.5, pose_track_conf=0.5,
        face_static=True,   # << important
        pose_static=True    # << important
    )
    phone = PhoneDetector(weights="yolov8n.pt", imgsz_1=416, conf_1=0.35,
                      imgsz_2=768, conf_2=0.25, iou=0.45, second_pass_gate=0.50, max_det=10)
    cap = open_cam(0, REQ_W, REQ_H, REQ_FPS)
    cv2.namedWindow("Multi-person Face+Hands", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Multi-person Face+Hands", 1100, 620)

    prev_t = time.time()
    fps_smooth = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # 1) Detect people
            persons = people_det.detect(frame)  # [(x1,y1,x2,y2,conf), ...]

            # 2) Per-person posture (face mesh + hands) inside each box
            overlay, people = mp_multi.process(frame, persons)

            # 3) Phones on top of the overlay (global)
            detections = phone.detect(frame)          # run on clean frame
            phone.draw(overlay, detections)           # draw onto overlay

            # 3) HUD
            if SHOW_FPS:
                now = time.time()
                dt = now - prev_t
                prev_t = now
                inst = (1.0 / dt) if dt > 0 else 0.0
                fps_smooth = inst if fps_smooth == 0 else (0.9 * fps_smooth + 0.1 * inst)
                cv2.putText(overlay, f"FPS: {fps_smooth:.1f}", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            cv2.imshow("Multi-person Face+Hands", overlay)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        mp_multi.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
