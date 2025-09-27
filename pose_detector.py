"""
StraightUp - Enhanced Pose Detection System
Real-time detection with MediaPipe and YOLO11n

Features:
- Eye contours and animated iris tracking
- Full 21-point hand skeleton with glow effects
- Complete body pose with neck center line mapping
- Enhanced phone detection with smooth tracking
- Real-time webcam processing with controls
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from ultralytics import YOLO
from collections import deque


class CompletePoseDetector:
    def __init__(self):
        # Initialize MediaPipe solutions
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize enhanced phone detection system
        print("Loading enhanced phone detection models...")
        self.yolo_model = YOLO('yolo11n.pt')  # Base COCO model
        print("✅ YOLO COCO model loaded (class 67: cell phone)")
        
        # Phone detection configuration
        self.phone_confidence_threshold = 0.25  # Lower threshold for better detection
        self.phone_classes = [67]  # COCO cell phone class
        self.phone_detection_history = []  # For tracking and smoothing
        self.max_history = 10
        
        # Enhanced phone tracking with smooth interpolation
        self._phone_tracks = []          # Active phone tracks
        self._next_track_id = 1
        self.phone_smooth_alpha = 0.6    # EMA smoothing factor
        self.phone_max_misses = 6        # Frames to keep tracks alive
        
        # EMA smoothing for neck center line
        self._neck_base_smooth = None
        self._head_pt_smooth = None
        
        print("📱 Enhanced phone detection with smooth tracking ready!")
        
        # Initialize models
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # MediaPipe landmark indices for detailed eye tracking
        self.LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        
        # Enhanced color scheme with neon/glow effects (BGR format)
        self.EYE_COLOR = (200, 200, 255)    # Light blue for eye contours
        self.IRIS_COLOR = (50, 150, 255)    # Warm orange for iris
        self.HAND_COLOR = (0, 255, 150)     # Bright green for hands
        self.SHOULDER_COLOR = (0, 255, 255)  # Yellow for shoulders
        self.POSE_COLOR = (255, 100, 0)     # Electric blue for body pose
        self.PHONE_COLOR = (255, 0, 255)    # Magenta for phones
        
        # Neon glow colors (brighter versions)
        self.GLOW_EYE = (255, 255, 100)
        self.GLOW_HAND = (100, 255, 200)
        self.GLOW_POSE = (255, 150, 50)
        self.GLOW_SHOULDER = (100, 255, 255)
        
        # Animation and trail variables
        self.prev_landmarks = None
        self.trail_points = []
        self.animation_frame = 0
        self.pulse_factor = 0
    
    # -------------------- Utility Functions --------------------
    def _ema(self, prev, cur, a=0.3):
        """Exponential Moving Average for smooth positioning"""
        if prev is None:
            return cur
        return (int(a*cur[0] + (1-a)*prev[0]), int(a*cur[1] + (1-a)*prev[1]))
        

    
    def draw_shoulder_highlight(self, image, landmarks):
        """Highlight shoulders specifically"""
        h, w = image.shape[:2]
        
        # Shoulder landmarks (11: left shoulder, 12: right shoulder)
        left_shoulder = landmarks.landmark[11]
        right_shoulder = landmarks.landmark[12]
        
        # Draw shoulder points
        left_x, left_y = int(left_shoulder.x * w), int(left_shoulder.y * h)
        right_x, right_y = int(right_shoulder.x * w), int(right_shoulder.y * h)
        
        # Draw highlighted shoulder points
        cv2.circle(image, (left_x, left_y), 10, self.SHOULDER_COLOR, -1)
        cv2.circle(image, (right_x, right_y), 10, self.SHOULDER_COLOR, -1)
        
        # Draw connecting line between shoulders
        cv2.line(image, (left_x, left_y), (right_x, right_y), self.SHOULDER_COLOR, 3)
        
        # Add shoulder labels
        cv2.putText(image, "L.SHOULDER", (left_x - 60, left_y - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.SHOULDER_COLOR, 1)
        cv2.putText(image, "R.SHOULDER", (right_x + 10, right_y - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.SHOULDER_COLOR, 1)
    
    def draw_glow_effect(self, image, points, color, glow_color, thickness=3):
        """Draw glowing lines with neon effect"""
        if len(points) < 2:
            return
        
        # Draw multiple layers for glow effect
        # Outer glow (thickest, most transparent)
        for i in range(len(points) - 1):
            cv2.line(image, points[i], points[i + 1], glow_color, thickness + 6)
        
        # Middle glow
        for i in range(len(points) - 1):
            cv2.line(image, points[i], points[i + 1], glow_color, thickness + 3)
        
        # Inner bright line
        for i in range(len(points) - 1):
            cv2.line(image, points[i], points[i + 1], color, thickness)
    
    def draw_neon_circle(self, image, center, radius, color, glow_color):
        """Draw a neon glowing circle"""
        # Outer glow
        cv2.circle(image, center, radius + 6, glow_color, -1)
        cv2.circle(image, center, radius + 3, glow_color, -1)
        # Inner bright circle
        cv2.circle(image, center, radius, color, -1)
        # Bright center
        cv2.circle(image, center, max(1, radius // 2), (255, 255, 255), -1)
    
    def draw_awesome_skeleton(self, image, landmarks):
        """Enhanced glowing skeleton with neck center line mapping (merged feature)"""
        h, w = image.shape[:2]
        
        # Get all landmark positions
        points = []
        for landmark in landmarks.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            points.append((x, y))
        
        # Define skeleton connections with different styles
        # Torso (main body) - Electric blue with thick glow
        torso_connections = [(11, 12), (11, 23), (12, 24), (23, 24)]  # shoulders to hips
        for start_idx, end_idx in torso_connections:
            if start_idx < len(points) and end_idx < len(points):
                self.draw_glow_effect(image, [points[start_idx], points[end_idx]], 
                                    self.POSE_COLOR, self.GLOW_POSE, 4)
        
        # Arms - Bright green with glow
        arm_connections = [(11, 13), (13, 15), (12, 14), (14, 16)]  # shoulders to wrists
        for start_idx, end_idx in arm_connections:
            if start_idx < len(points) and end_idx < len(points):
                self.draw_glow_effect(image, [points[start_idx], points[end_idx]], 
                                    self.HAND_COLOR, self.GLOW_HAND, 3)
        
        # Legs - Purple with glow
        leg_connections = [(23, 25), (25, 27), (24, 26), (26, 28)]  # hips to ankles
        leg_color = (255, 100, 255)  # Purple
        glow_leg = (255, 150, 255)   # Light purple glow
        for start_idx, end_idx in leg_connections:
            if start_idx < len(points) and end_idx < len(points):
                self.draw_glow_effect(image, [points[start_idx], points[end_idx]], 
                                    leg_color, glow_leg, 3)
        
        # NECK LINES AND CENTER MAPPING
        if len(points) > 12:
            # Shoulder→Head (nose) lines
            self.draw_glow_effect(image, [points[11], points[0]],
                                  self.SHOULDER_COLOR, self.GLOW_SHOULDER, 3)
            self.draw_glow_effect(image, [points[12], points[0]],
                                  self.SHOULDER_COLOR, self.GLOW_SHOULDER, 3)

            # CENTER NECK LINE: mid-shoulders -> head anchor
            mx = int((points[11][0] + points[12][0]) / 2)
            my = int((points[11][1] + points[12][1]) / 2)
            neck_base_raw = (mx, my)
            head_pt_raw = points[0]

            # Apply EMA smoothing for silky smooth neck line
            neck_base = self._ema(self._neck_base_smooth, neck_base_raw, a=0.7) if self._neck_base_smooth else neck_base_raw
            head_pt   = self._ema(self._head_pt_smooth,   head_pt_raw,   a=0.7) if self._head_pt_smooth   else head_pt_raw
            self._neck_base_smooth = neck_base
            self._head_pt_smooth   = head_pt

            # Draw the crisp, smooth center neck line
            cv2.line(image, neck_base, head_pt, self.SHOULDER_COLOR, 2, lineType=cv2.LINE_AA)
            self.draw_neon_circle(image, neck_base, 8, self.SHOULDER_COLOR, self.GLOW_SHOULDER)
            cv2.putText(image, "NECK", (neck_base[0] - 20, neck_base[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.SHOULDER_COLOR, 1)
        
        # Draw glowing joint points
        joint_indices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]  # Major joints
        for idx in joint_indices:
            if idx < len(points):
                # Pulsing effect
                pulse = int(10 + 5 * np.sin(self.animation_frame * 0.2 + idx))
                joint_color = self.POSE_COLOR
                if idx in [11, 12]:  # Shoulders get special treatment
                    joint_color = self.SHOULDER_COLOR
                    self.draw_neon_circle(image, points[idx], pulse, joint_color, self.GLOW_SHOULDER)
                else:
                    self.draw_neon_circle(image, points[idx], pulse, joint_color, self.GLOW_POSE)
    
    def draw_awesome_hands(self, image, hand_landmarks):
        """Draw awesome glowing hands with particle effects"""
        h, w = image.shape[:2]
        
        # Get hand points
        hand_points = []
        for landmark in hand_landmarks.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            hand_points.append((x, y))
        
        # Draw glowing hand connections
        connections = [
            # Thumb
            [0, 1, 2, 3, 4],
            # Index finger  
            [0, 5, 6, 7, 8],
            # Middle finger
            [0, 9, 10, 11, 12],
            # Ring finger
            [0, 13, 14, 15, 16],
            # Pinky
            [0, 17, 18, 19, 20]
        ]
        
        # Draw each finger with glow
        for finger in connections:
            finger_points = [hand_points[i] for i in finger if i < len(hand_points)]
            if len(finger_points) > 1:
                self.draw_glow_effect(image, finger_points, self.HAND_COLOR, self.GLOW_HAND, 2)
        
        # Draw glowing fingertip points
        fingertips = [4, 8, 12, 16, 20]  # Tip of each finger
        for tip_idx in fingertips:
            if tip_idx < len(hand_points):
                # Animated glow for fingertips
                pulse = int(8 + 3 * np.sin(self.animation_frame * 0.3 + tip_idx))
                self.draw_neon_circle(image, hand_points[tip_idx], pulse, 
                                    self.HAND_COLOR, self.GLOW_HAND)
    
    def draw_awesome_eyes(self, image, landmarks, eye_indices, iris_indices):
        """Draw cool but natural-looking eyes with subtle effects"""
        h, w = image.shape[:2]
        
        # Draw eye contour with subtle glow
        eye_points = []
        for idx in eye_indices:
            if idx < len(landmarks.landmark):
                x = int(landmarks.landmark[idx].x * w)
                y = int(landmarks.landmark[idx].y * h)
                eye_points.append((x, y))
                # Draw small eye contour points
                cv2.circle(image, (x, y), 1, self.EYE_COLOR, -1)
        
        if len(eye_points) > 3:
            # Draw subtle eye outline
            eye_poly = np.array(eye_points, dtype=np.int32)
            cv2.polylines(image, [eye_poly], True, (150, 200, 255), 2)  # Subtle outer glow
            cv2.polylines(image, [eye_poly], True, self.EYE_COLOR, 1)   # Main line
        
        # Draw iris with gentle animation
        iris_points = []
        for idx in iris_indices:
            if idx < len(landmarks.landmark):
                x = int(landmarks.landmark[idx].x * w)
                y = int(landmarks.landmark[idx].y * h)
                iris_points.append((x, y))
        
        if len(iris_points) >= 4:
            iris_center = np.mean(iris_points, axis=0).astype(int)
            # Gentle iris glow - less intense
            base_radius = 8
            pulse = int(base_radius + 2 * np.sin(self.animation_frame * 0.1))
            
            # Subtle iris glow
            cv2.circle(image, tuple(iris_center), pulse + 3, (100, 100, 255), -1)  # Soft glow
            cv2.circle(image, tuple(iris_center), pulse, self.IRIS_COLOR, -1)      # Main iris
            cv2.circle(image, tuple(iris_center), max(1, pulse // 3), (255, 255, 255), -1)  # Pupil
    
    # -------------------- Enhanced Phone Detection --------------------
    def _iou(self, a, b):
        """Calculate Intersection over Union for box overlap detection"""
        ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a + area_b - inter + 1e-6)

    def _lerp_bbox(self, a, b, alpha):
        """Linear interpolation between two bounding boxes for smooth tracking"""
        ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
        nx1 = int((1 - alpha) * ax1 + alpha * bx1)
        ny1 = int((1 - alpha) * ay1 + alpha * by1)
        nx2 = int((1 - alpha) * ax2 + alpha * bx2)
        ny2 = int((1 - alpha) * ay2 + alpha * by2)
        return (nx1, ny1, nx2, ny2)

    def _detect_phones_all_sources(self, image, hand_landmarks=None):
        """Enhanced phone detection with smooth tracking"""
        dets = []

        # COCO cell phone detection
        results = self.yolo_model(image, verbose=False)
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(box.cls)
                conf = float(box.conf)
                if class_id == 67 and conf > self.phone_confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    dets.append({"bbox": (x1, y1, x2, y2), "confidence": conf, "type": "COCO"})

        return dets

    def _update_phone_tracks_and_get_boxes(self, detections):
        """Match detections to tracks, smooth with EMA, keep for a few misses"""
        alpha = self.phone_smooth_alpha
        iou_thr = 0.3

        unmatched_tracks = set(range(len(self._phone_tracks)))

        # Match detections to existing tracks (greedy IoU)
        for det in detections:
            best_iou, best_idx = 0.0, -1
            for ti in unmatched_tracks:
                iou = self._iou(self._phone_tracks[ti]['bbox'], det['bbox'])
                if iou > best_iou:
                    best_iou, best_idx = iou, ti
            if best_iou >= iou_thr and best_idx != -1:
                t = self._phone_tracks[best_idx]
                t['bbox'] = self._lerp_bbox(t['bbox'], det['bbox'], alpha)
                t['type'] = det['type']
                t['conf'] = det['confidence']
                t['misses'] = 0
                unmatched_tracks.discard(best_idx)
            else:
                # New track
                self._phone_tracks.append({
                    "id": self._next_track_id,
                    "bbox": det['bbox'],
                    "type": det['type'],
                    "conf": det['confidence'],
                    "misses": 0
                })
                self._next_track_id += 1

        # Age out unmatched tracks
        kept = []
        for idx, t in enumerate(self._phone_tracks):
            if idx in unmatched_tracks:
                t['misses'] += 1
            if t['misses'] <= self.phone_max_misses:
                kept.append(t)
        self._phone_tracks = kept

        # Return smoothed boxes to draw
        return [{"bbox": t["bbox"], "confidence": t.get("conf", 0.0), "type": t.get("type", "COCO")}
                for t in self._phone_tracks]

    def _draw_enhanced_phone_detections(self, image, phone_boxes):
        """Draw smoothed phone boxes with enhanced visualization"""
        for phone in phone_boxes:
            x1, y1, x2, y2 = phone['bbox']
            conf = phone.get('confidence', 0.0)
            det_type = phone.get('type', 'COCO')

            if det_type == 'COCO':
                color = self.PHONE_COLOR; label_prefix = "📱PHONE"
            else:
                color = (0, 255, 255); label_prefix = "📱OTHER"

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)
            cv2.rectangle(image, (x1-2, y1-2), (x2+2, y2+2), color, 1)
            label = f"{label_prefix} {conf:.2f}"
            sz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(image, (x1, y1 - sz[1] - 12), (x1 + sz[0] + 10, y1 - 2), color, -1)
            cv2.putText(image, label, (x1 + 5, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def detect_phones_enhanced(self, image, hand_landmarks=None):
        """Enhanced phone detection with smooth tracking (merged and simplified)"""
        # Detect phones EVERY frame -> smooth -> draw
        hand_lms_list = hand_landmarks if hand_landmarks else None
        raw_phone_dets = self._detect_phones_all_sources(image, hand_lms_list)
        smoothed_boxes = self._update_phone_tracks_and_get_boxes(raw_phone_dets)
        self._draw_enhanced_phone_detections(image, smoothed_boxes)
        phones_detected_now = len([t for t in self._phone_tracks if t['misses'] == 0])
        
        # Update detection history for smoothing
        self.phone_detection_history.append(phones_detected_now)
        if len(self.phone_detection_history) > self.max_history:
            self.phone_detection_history.pop(0)
        
        return phones_detected_now
    
    def process_frame(self, image):
        """Process one frame: MediaPipe detection, smooth phone tracking, draw all features"""
        # Convert once
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # MediaPipe processing
        face_results = self.face_mesh.process(rgb_image)
        hand_results = self.hands.process(rgb_image)
        pose_results = self.pose.process(rgb_image)

        faces_detected = len(face_results.multi_face_landmarks) if getattr(face_results, "multi_face_landmarks", None) else 0
        hands_detected = len(hand_results.multi_hand_landmarks) if getattr(hand_results, "multi_hand_landmarks", None) else 0
        pose_detected  = 1 if getattr(pose_results, "pose_landmarks", None) is not None else 0

        # Enhanced phone detection with smooth tracking
        hand_lms_list = hand_results.multi_hand_landmarks if hands_detected else None
        phones_detected = self.detect_phones_enhanced(image, hand_lms_list)

        # Animation tick
        self.animation_frame += 1

        # Draw face mesh + eyes
        if faces_detected:
            for face_lms in face_results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_lms,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(60, 60, 60), thickness=1)
                )
                self.draw_awesome_eyes(image, face_lms, self.LEFT_EYE, self.LEFT_IRIS)
                self.draw_awesome_eyes(image, face_lms, self.RIGHT_EYE, self.RIGHT_IRIS)

        # Draw hands
        if hands_detected:
            for hand_lms in hand_results.multi_hand_landmarks:
                self.draw_awesome_hands(image, hand_lms)

        # Draw pose + shoulders + neck line (KEY MERGED FEATURE!)
        if pose_detected:
            self.draw_awesome_skeleton(image, pose_results.pose_landmarks)
            self.draw_shoulder_highlight(image, pose_results.pose_landmarks)

        return image, faces_detected, hands_detected, pose_detected, phones_detected
    
    def run_webcam(self):
        """Run the complete detection on webcam"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("🎯 StraightUp Detection System Started!")
        print("=" * 50)
        print("✨ Visual Features:")
        print("   👁️  Eyes: Natural contours + animated iris")
        print("   🤲 Hands: NEON skeleton with pulsing fingertips")
        print("   💪 Shoulders: HIGHLIGHTED with glow effects")
        print("   🏃 Body: ELECTRIC skeleton with animated joints")
        print("   � Neck: Smooth center line mapping (EMA smoothed)")
        print("   � Phones: Enhanced YOLO11n tracking with smoothing")
        print("   ✨ All with GLOW EFFECTS and smooth animations!")
        print("\n⌨️  Controls:")
        print("   'q' or ESC: Quit application")
        print("   's': Save current frame")
        print("   'i': Toggle info display")
        print("   Space: Pause/Resume detection")
        print("=" * 50)
        
        # Performance tracking
        prev_time = time.time()
        fps_counter = 0
        show_info = True
        paused = False
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("Failed to grab frame")
                        break
                    
                    # Flip frame horizontally for selfie-view
                    frame = cv2.flip(frame, 1)
                    
                    # Process detections
                    processed_frame, faces, hands, pose, phones = self.process_frame(frame)
                else:
                    processed_frame = frame.copy()
                
                # Calculate FPS
                current_time = time.time()
                fps_counter += 1
                if current_time - prev_time >= 1.0:
                    fps = fps_counter / (current_time - prev_time)
                    fps_counter = 0
                    prev_time = current_time
                else:
                    fps = 0
                
                # Draw info overlay
                if show_info:
                    h, w = processed_frame.shape[:2]
                    
                    # Status box
                    cv2.rectangle(processed_frame, (10, 10), (400, 145), (0, 0, 0), -1)
                    cv2.rectangle(processed_frame, (10, 10), (400, 145), (255, 255, 255), 2)
                    
                    # Status text
                    cv2.putText(processed_frame, f"FPS: {fps:.1f}", (20, 35), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(processed_frame, f"Faces: {faces}", (20, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(processed_frame, f"Hands: {hands}", (20, 85), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(processed_frame, f"Pose: {'Yes' if pose else 'No'}", (20, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.putText(processed_frame, f"Phones: {phones}", (20, 135), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                    
                    # Legend
                    legend_y = h - 80
                    cv2.rectangle(processed_frame, (10, legend_y - 10), (450, h - 10), (0, 0, 0), -1)
                    cv2.rectangle(processed_frame, (10, legend_y - 10), (450, h - 10), (255, 255, 255), 1)
                    
                    cv2.putText(processed_frame, "Eyes: Cyan + Red Iris", (20, legend_y + 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    cv2.putText(processed_frame, "Hands: Green", (20, legend_y + 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(processed_frame, "Shoulders: Yellow", (20, legend_y + 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    cv2.putText(processed_frame, "Body: Blue/Pink", (250, legend_y + 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    cv2.putText(processed_frame, "Phones: Magenta (Enhanced Tracking)", (250, legend_y + 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                    cv2.putText(processed_frame, "Neck: Yellow center line (EMA smooth)", (250, legend_y + 45), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
                # Pause indicator
                if paused:
                    h, w = processed_frame.shape[:2]
                    cv2.rectangle(processed_frame, (w//2 - 100, h//2 - 30), (w//2 + 100, h//2 + 30), (0, 0, 0), -1)
                    cv2.putText(processed_frame, "PAUSED", (w//2 - 80, h//2 + 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                
                cv2.imshow('StraightUp - Enhanced Pose Detection', processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord('s'):  # Save frame
                    timestamp = int(time.time())
                    filename = f'detection_capture_{timestamp}.jpg'
                    cv2.imwrite(filename, processed_frame)
                    print(f"💾 Frame saved as {filename}")
                elif key == ord('i'):  # Toggle info
                    show_info = not show_info
                    print(f"ℹ️  Info display: {'ON' if show_info else 'OFF'}")
                elif key == ord(' '):  # Space to pause
                    paused = not paused
                    print(f"⏸️  {'Paused' if paused else 'Resumed'}")
        
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Detection completed")

def main():
    """Main function to run the complete detection system"""
    print("🚀 Starting MediaPipe Complete Detection System...")
    detector = CompletePoseDetector()
    detector.run_webcam()


if __name__ == "__main__":
    main()