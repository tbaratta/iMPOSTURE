# multiperson_posture.py  — multi-face + multi-hands + per-person pose
import cv2
import math
import mediapipe as mp

mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh      = mp.solutions.face_mesh
mp_hands          = mp.solutions.hands
mp_pose           = mp.solutions.pose

def _dist(a, b):
    return math.hypot(b[0]-a[0], b[1]-a[1])

def _expand_and_clamp(x1, y1, x2, y2, w, h, pad=0.12):
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    bw = (x2 - x1)
    bh = (y2 - y1)
    bw2 = int(bw * (1 + pad))
    bh2 = int(bh * (1 + pad))
    nx1 = max(0, int(cx - bw2 / 2))
    ny1 = max(0, int(cy - bh2 / 2))
    nx2 = min(w-1, int(cx + bw2 / 2))
    ny2 = min(h-1, int(cy + bh2 / 2))
    return nx1, ny1, nx2, ny2

def _make_square(img):
    h, w = img.shape[:2]
    side = max(h, w)
    top = (side - h) // 2
    bottom = side - h - top
    left = (side - w) // 2
    right = side - w - left
    sq = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))
    return sq, (top, left, h, w)

class MultiPersonPosture:
    def __init__(self,
                 max_faces_per_person=1,
                 max_hands_per_person=2,
                 face_detect_conf=0.5,
                 face_track_conf=0.5,
                 hand_detect_conf=0.5,
                 hand_track_conf=0.5,
                 pose_detect_conf=0.5,
                 pose_track_conf=0.5,
                 face_static=True,   # << force fresh detection per ROI
                 pose_static=True):  # << force fresh detection per ROI
        # FaceMesh (STATIC so each ROI gets its own detection)
        self.fm = mp_face_mesh.FaceMesh(
            static_image_mode=face_static,
            max_num_faces=max_faces_per_person,
            refine_landmarks=True,
            min_detection_confidence=face_detect_conf,
            min_tracking_confidence=face_track_conf
        )
        # Hands (track is fine across frames; we call per ROI)
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands_per_person,
            min_detection_confidence=hand_detect_conf,
            min_tracking_confidence=hand_track_conf
        )
        # Pose (STATIC so each ROI gets its own detection)
        self.pose = mp_pose.Pose(
            static_image_mode=pose_static,
            model_complexity=1,
            enable_segmentation=False,
            smooth_landmarks=True,
            min_detection_confidence=pose_detect_conf,
            min_tracking_confidence=pose_track_conf
        )

        self.hand_lm_spec   = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)
        self.hand_conn_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)

        HL = mp_hands.HandLandmark
        self.HL = HL
        self.fingertips = [HL.THUMB_TIP, HL.INDEX_FINGER_TIP, HL.MIDDLE_FINGER_TIP,
                           HL.RING_FINGER_TIP, HL.PINKY_TIP]
        self.finger_names = {
            HL.THUMB_TIP: "Thumb",
            HL.INDEX_FINGER_TIP: "Index",
            HL.MIDDLE_FINGER_TIP: "Middle",
            HL.RING_FINGER_TIP: "Ring",
            HL.PINKY_TIP: "Pinky"
        }

    def close(self):
        self.fm.close()
        self.hands.close()
        self.pose.close()

    def _hand_names_px(self, hand_lm, W, H):
        return {idx.name: (int(lm.x * W), int(lm.y * H))
                for idx, lm in zip(self.HL, hand_lm.landmark)}

    def _fingers_up(self, names_px, handed_label):
        HL = self.HL
        tip_ids = {
            "thumb": HL.THUMB_TIP,
            "index": HL.INDEX_FINGER_TIP,
            "middle": HL.MIDDLE_FINGER_TIP,
            "ring": HL.RING_FINGER_TIP,
            "pinky": HL.PINKY_TIP
        }
        pip_ids = {
            "index": HL.INDEX_FINGER_PIP,
            "middle": HL.MIDDLE_FINGER_PIP,
            "ring": HL.RING_FINGER_PIP,
            "pinky": HL.PINKY_PIP
        }
        up = {}
        tx, _ = names_px[HL.THUMB_TIP.name]
        ipx, _ = names_px[HL.THUMB_IP.name]
        is_right = handed_label.lower().startswith("right")
        up["thumb"] = (tx > ipx) if is_right else (tx < ipx)
        for f in ["index","middle","ring","pinky"]:
            tip_y = names_px[tip_ids[f].name][1]
            pip_y = names_px[pip_ids[f].name][1]
            up[f] = tip_y < pip_y
        cnt = sum(1 for v in up.values() if v)
        return up, cnt

    def _posture_from_pose(self, lm, W, H):
        LSH = mp_pose.PoseLandmark.LEFT_SHOULDER.value
        RSH = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        LEAR= mp_pose.PoseLandmark.LEFT_EAR.value
        REAR= mp_pose.PoseLandmark.RIGHT_EAR.value
        def _p(i): return (int(lm[i].x * W), int(lm[i].y * H))
        lsh, rsh = _p(LSH), _p(RSH)
        # neck & head proxy
        neck = ((lsh[0] + rsh[0]) // 2, (lsh[1] + rsh[1]) // 2)
        lear, rear = _p(LEAR), _p(REAR)
        head = ((lear[0] + rear[0]) // 2, (lear[1] + rear[1]) // 2)
        # angles
        def _angle_deg(a, b): return math.degrees(math.atan2(b[1]-a[1], b[0]-a[0]))
        sh_angle = _angle_deg(lsh, rsh)
        if sh_angle > 90: sh_angle -= 180
        if sh_angle < -90: sh_angle += 180
        neck_vec = _angle_deg(neck, head)
        neck_verticality = 90 - abs(neck_vec - 90)
        head_forward_px = abs(head[0] - neck[0])
        return {
            "lsh": lsh, "rsh": rsh, "neck": neck, "head": head,
            "shoulder_slope_deg": sh_angle,
            "neck_verticality_deg": neck_verticality,
            "head_forward_px": head_forward_px
        }

    def process(self, frame_bgr, person_boxes):
        """
        Args:
          frame_bgr: BGR frame
          person_boxes: [(x1,y1,x2,y2, conf), ...]
        Returns:
          overlay_bgr, people (metrics list per person)
        """
        base = frame_bgr.copy()
        H0, W0 = base.shape[:2]
        people = []

        for pid, (x1,y1,x2,y2,pc) in enumerate(person_boxes):
            x1e,y1e,x2e,y2e = _expand_and_clamp(x1,y1,x2,y2,W0,H0,pad=0.15)
            roi = base[y1e:y2e, x1e:x2e]
            if roi.size == 0:
                continue

            sq, (top,left,h_roi,w_roi) = _make_square(roi)

            # Run all MP modules on the square ROI
            rgb = cv2.cvtColor(sq, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            face_res  = self.fm.process(rgb)   # static_image_mode=True -> fresh detection per ROI
            hands_res = self.hands.process(rgb)
            pose_res  = self.pose.process(rgb) # static_image_mode=True -> fresh detection per ROI
            rgb.flags.writeable = True
            out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            H, W = out.shape[:2]

            person = {"id": pid, "person_conf": pc, "faces": [], "hands": [], "pose": None}

            # ---- Face meshes (allow 1 per person; you can set >1 if needed) ----
            if face_res and face_res.multi_face_landmarks:
                for fl in face_res.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=out, landmark_list=fl,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    mp_drawing.draw_landmarks(
                        image=out, landmark_list=fl,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                    )
                    mp_drawing.draw_landmarks(
                        image=out, landmark_list=fl,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                    )
                    r_eye = fl.landmark[33]; l_eye = fl.landmark[263]
                    p1 = (int(r_eye.x * W), int(r_eye.y * H))
                    p2 = (int(l_eye.x * W), int(l_eye.y * H))
                    inter_eye = _dist(p1, p2)
                    person["faces"].append({"inter_eye_px": inter_eye})

            # ---- Pose skeleton per person ----
            if pose_res and pose_res.pose_landmarks:
                mp_drawing.draw_landmarks(
                    out, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(thickness=2)
                )
                person["pose"] = self._posture_from_pose(pose_res.pose_landmarks.landmark, W, H)
                # draw shoulder/neck lines
                if person["pose"] is not None:
                    lsh = person["pose"]["lsh"]; rsh = person["pose"]["rsh"]
                    neck = person["pose"]["neck"]; head = person["pose"]["head"]
                    cv2.line(out, lsh, rsh, (0,255,255), 2)
                    cv2.line(out, neck, head, (255,200,0), 2)

            # ---- Hands with fingertips + fingers-up ----
            if hands_res and hands_res.multi_hand_landmarks:
                # Use handedness if available; else fallback label
                handed_list = hands_res.multi_handedness or [None]*len(hands_res.multi_hand_landmarks)
                for hand_lm, handed in zip(hands_res.multi_hand_landmarks, handed_list):
                    label = handed.classification[0].label if handed else "Hand"
                    score = handed.classification[0].score if handed else 1.0

                    mp_drawing.draw_landmarks(
                        out, hand_lm, mp_hands.HAND_CONNECTIONS,
                        self.hand_lm_spec, self.hand_conn_spec
                    )
                    for tip_id in self.fingertips:
                        tip = hand_lm.landmark[tip_id]
                        cx, cy = int(tip.x * W), int(tip.y * H)
                        cv2.circle(out, (cx, cy), 6, (0, 255, 0), -1)

                    names_px = self._hand_names_px(hand_lm, W, H)
                    fingers, cnt = self._fingers_up(names_px, label)
                    wx, wy = names_px[self.HL.WRIST.name]
                    cv2.putText(out, f"{label}: {cnt}", (wx, max(0, wy-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)

                    person["hands"].append({
                        "label": label,
                        "score": float(score),
                        "fingers_up_count": int(cnt),
                        "fingers_up": fingers
                    })

            # paste ROI overlay back to full frame
            overlay_roi = out[top:top+h_roi, left:left+w_roi]
            base[y1e:y2e, x1e:x2e] = overlay_roi

            # draw YOLO person box & id
            cv2.rectangle(base, (x1, y1), (x2, y2), (255, 120, 0), 2)
            cv2.putText(base, f"ID {pid}", (x1, max(0, y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,120,0), 2)

            people.append(person)

        return base, people
