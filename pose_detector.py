"""
Complete Face, Hand, and Pose Detection with MediaPipe
Real-time detection of eyes, hands, and shoulders using Google's MediaPipe

Features:
- Detailed eye contours and iris tracking
- Full 21-point hand skeleton detection
- Complete body pose with shoulder highlighting
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
        
        # TRUE MUID-IITR parameters
        self.enable_hand_phone_fusion = True  # Combine hand + phone detection
        self.phone_near_hand_threshold = 100  # pixels
        self.compound_detection_threshold = 120  # Distance for compound boxes
        self.usage_confidence_threshold = 0.5    # Minimum usage confidence
        self.muid_detection_history = deque(maxlen=10)  # Usage history for smoothing
        
        print("📱 TRUE MUID-IITR phone usage detection ready!")
        print("   ✅ Compound bounding boxes enabled")
        print("   ✅ Usage scenario detection active")
        print("   ✅ Hand-phone interaction analysis ready")
        
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
        
    def draw_eye_landmarks_old(self, image, landmarks, eye_indices, iris_indices):
        """Draw detailed eye landmarks including iris"""
        h, w = image.shape[:2]
        
        # Draw eye contour
        eye_points = []
        for idx in eye_indices:
            if idx < len(landmarks.landmark):
                x = int(landmarks.landmark[idx].x * w)
                y = int(landmarks.landmark[idx].y * h)
                eye_points.append((x, y))
                cv2.circle(image, (x, y), 2, self.EYE_COLOR, -1)
        
        # Draw eye contour line
        if len(eye_points) > 3:
            eye_points = np.array(eye_points, dtype=np.int32)
            cv2.polylines(image, [eye_points], True, self.EYE_COLOR, 1)
        
        # Draw iris
        iris_points = []
        for idx in iris_indices:
            if idx < len(landmarks.landmark):
                x = int(landmarks.landmark[idx].x * w)
                y = int(landmarks.landmark[idx].y * h)
                iris_points.append((x, y))
                cv2.circle(image, (x, y), 3, self.IRIS_COLOR, -1)
        
        # Draw iris circle
        if len(iris_points) >= 4:
            iris_center = np.mean(iris_points, axis=0).astype(int)
            iris_radius = int(np.linalg.norm(np.array(iris_points[0]) - iris_center))
            cv2.circle(image, tuple(iris_center), iris_radius, self.IRIS_COLOR, 2)
    
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
        """Draw an awesome glowing skeleton with effects"""
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
    
    def detect_phones_enhanced(self, image, hand_landmarks=None):
        """Enhanced phone detection using multiple approaches"""
        phones_detected = 0
        phone_boxes = []
        
        # Method 1: COCO Dataset Detection (class 67)
        results = self.yolo_model(image, verbose=False)
        coco_phones = self._detect_coco_phones(results, image)
        phones_detected += len(coco_phones)
        phone_boxes.extend(coco_phones)
        
        # Method 2: TRUE MUID-IITR Compound Detection
        compound_detections = self._detect_compound_usage_boxes(image, hand_landmarks)
        phones_detected += len(compound_detections)
        phone_boxes.extend(compound_detections)
        
        # Update detection history for smoothing
        self.muid_detection_history.append(len(compound_detections))
        
        # Method 3: Hand-Phone Fusion (detect phones near hands)
        if self.enable_hand_phone_fusion and hand_landmarks:
            fusion_phones = self._detect_phones_near_hands(image, hand_landmarks, phone_boxes)
            phones_detected += len(fusion_phones)
            phone_boxes.extend(fusion_phones)
        
        # Remove duplicate detections
        phone_boxes = self._remove_duplicate_detections(phone_boxes)
        phones_detected = len(phone_boxes)
        
        # Draw all detected phones with enhanced visualization
        self._draw_enhanced_phone_detections(image, phone_boxes)
        
        # Update detection history for smoothing
        self.phone_detection_history.append(phones_detected)
        if len(self.phone_detection_history) > self.max_history:
            self.phone_detection_history.pop(0)
        
        return phones_detected
    
    def _detect_coco_phones(self, results, image):
        """Detect phones using standard COCO dataset"""
        phones = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls)
                    confidence = float(box.conf)
                    
                    # COCO cell phone detection with lower threshold
                    if class_id == 67 and confidence > self.phone_confidence_threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        phones.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': confidence,
                            'type': 'COCO',
                            'class_id': class_id
                        })
        
        return phones
    
    def _detect_compound_usage_boxes(self, image, hand_landmarks):
        """TRUE MUID-IITR: Detect compound bounding boxes for actual phone usage"""
        if not hand_landmarks:
            return []
        
        h, w = image.shape[:2]
        
        # Step 1: Get basic phone detections
        results = self.yolo_model(image, verbose=False)
        basic_phones = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls)
                    confidence = float(box.conf)
                    
                    if class_id == 67 and confidence > 0.3:  # Only cell phones
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        basic_phones.append({
                            'bbox': (x1, y1, x2, y2),
                            'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                            'confidence': confidence
                        })
        
        # Step 2: Get hand positions from MediaPipe
        hand_positions = []
        for hand_lm in hand_landmarks:
            landmarks = []
            for landmark in hand_lm.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                landmarks.append((x, y))
            
            # Calculate hand center and bounding box
            min_x = min([p[0] for p in landmarks])
            max_x = max([p[0] for p in landmarks])
            min_y = min([p[1] for p in landmarks])
            max_y = max([p[1] for p in landmarks])
            center_x = (min_x + max_x) // 2
            center_y = (min_y + max_y) // 2
            
            hand_positions.append({
                'bbox': (min_x, min_y, max_x, max_y),
                'center': (center_x, center_y),
                'landmarks': landmarks,
                'fingertips': [landmarks[4], landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
            })
        
        # Step 3: Create compound detections (CORE MUID-IITR FEATURE)
        compound_detections = []
        for hand in hand_positions:
            for phone in basic_phones:
                distance = self._calculate_distance(hand['center'], phone['center'])
                
                if distance < self.compound_detection_threshold:
                    # Create compound bounding box
                    hand_bbox = hand['bbox']
                    phone_bbox = phone['bbox']
                    
                    compound_x1 = min(hand_bbox[0], phone_bbox[0])
                    compound_y1 = min(hand_bbox[1], phone_bbox[1])
                    compound_x2 = max(hand_bbox[2], phone_bbox[2])
                    compound_y2 = max(hand_bbox[3], phone_bbox[3])
                    
                    # Calculate usage confidence
                    usage_confidence = self._calculate_usage_confidence(
                        hand, phone, distance, image
                    )
                    
                    if usage_confidence > self.usage_confidence_threshold:
                        compound_detections.append({
                            'compound_bbox': (compound_x1, compound_y1, compound_x2, compound_y2),
                            'hand_bbox': hand_bbox,
                            'phone_bbox': phone_bbox,
                            'usage_confidence': usage_confidence,
                            'distance': distance,
                            'type': 'COMPOUND_USAGE',
                            'hand_center': hand['center'],
                            'phone_center': phone['center']
                        })
        
        return compound_detections
    
    def _calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        import math
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _calculate_usage_confidence(self, hand, phone, distance, image):
        """Calculate confidence that this represents actual phone usage"""
        h, w = image.shape[:2]
        
        # Base confidence from proximity
        proximity_score = 1.0 - (distance / self.compound_detection_threshold)
        
        # Bonus if phone is in upper portion (near face/ear)
        phone_y = phone['center'][1]
        face_area_bonus = 0.3 if phone_y < h * 0.6 else 0.0
        
        # Hand orientation analysis
        orientation_score = self._analyze_hand_phone_orientation(hand, phone)
        
        # Size validation (phones shouldn't be too big or small in usage)
        phone_bbox = phone['bbox']
        phone_area = (phone_bbox[2] - phone_bbox[0]) * (phone_bbox[3] - phone_bbox[1])
        relative_area = phone_area / (h * w)
        size_penalty = 0.2 if relative_area > 0.15 or relative_area < 0.005 else 0.0
        
        usage_confidence = min(1.0, proximity_score + face_area_bonus + orientation_score - size_penalty)
        return max(0.0, usage_confidence)
    
    def _analyze_hand_phone_orientation(self, hand, phone):
        """Analyze if hand positioning suggests phone usage"""
        fingertips = hand['fingertips']
        phone_center = phone['center']
        
        # Count fingertips near phone
        tips_near_phone = 0
        for tip in fingertips:
            tip_phone_distance = self._calculate_distance(tip, phone_center)
            if tip_phone_distance < 80:
                tips_near_phone += 1
        
        # More fingertips near phone = higher usage likelihood
        orientation_score = (tips_near_phone / 5) * 0.3
        
        # Check if hand is "wrapped around" phone (usage position)
        hand_center = hand['center']
        hand_phone_vector = (phone_center[0] - hand_center[0], phone_center[1] - hand_center[1])
        
        # Bonus for typical holding positions
        if abs(hand_phone_vector[0]) < 50 and hand_phone_vector[1] < 0:  # Phone above hand
            orientation_score += 0.2
        
        return min(0.4, orientation_score)
    
    def _detect_phones_near_hands(self, image, hand_landmarks, existing_phones):
        """Detect potential phones near hand positions"""
        phones = []
        h, w = image.shape[:2]
        
        if not hand_landmarks:
            return phones
        
        # Get hand positions
        hand_positions = []
        for hand_lm in hand_landmarks:
            for landmark in hand_lm.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                hand_positions.append((x, y))
        
        # Look for rectangular objects near hands
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 1000 < area < 20000:  # Reasonable phone-sized area
                # Get bounding rectangle
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                
                # Check if it's near any hand
                contour_center = (x + w_rect//2, y + h_rect//2)
                near_hand = any(
                    np.sqrt((contour_center[0] - hx)**2 + (contour_center[1] - hy)**2) < self.phone_near_hand_threshold
                    for hx, hy in hand_positions
                )
                
                if near_hand:
                    # Check if it's not already detected
                    contour_bbox = (x, y, x + w_rect, y + h_rect)
                    is_duplicate = any(
                        self._boxes_overlap(contour_bbox, 
                                          phone['compound_bbox'] if phone['type'] == 'COMPOUND_USAGE' 
                                          else phone['bbox'])
                        for phone in existing_phones
                    )
                    
                    if not is_duplicate:
                        phones.append({
                            'bbox': (x, y, x + w_rect, y + h_rect),
                            'confidence': 0.6,  # Medium confidence for fusion detection
                            'type': 'Hand-Fusion',
                            'class_id': 67
                        })
        
        return phones
    
    def _remove_duplicate_detections(self, phone_boxes):
        """Remove overlapping phone detections"""
        if len(phone_boxes) <= 1:
            return phone_boxes
        
        # Sort by confidence (handle different confidence key names)
        def get_confidence(detection):
            if 'usage_confidence' in detection:
                return detection['usage_confidence']
            elif 'confidence' in detection:
                return detection['confidence']
            else:
                return 0.0
        
        phone_boxes.sort(key=get_confidence, reverse=True)
        
        filtered_phones = []
        for phone in phone_boxes:
            # Get the bounding box for overlap detection
            if phone['type'] == 'COMPOUND_USAGE':
                phone_bbox = phone['compound_bbox']
            else:
                phone_bbox = phone['bbox']
            
            is_duplicate = any(
                self._boxes_overlap(phone_bbox, 
                                  existing['compound_bbox'] if existing['type'] == 'COMPOUND_USAGE' 
                                  else existing['bbox'])
                for existing in filtered_phones
            )
            if not is_duplicate:
                filtered_phones.append(phone)
        
        return filtered_phones
    
    def _boxes_overlap(self, box1, box2, threshold=0.3):
        """Check if two bounding boxes overlap significantly"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return False
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        iou = intersection / (area1 + area2 - intersection)
        return iou > threshold
    
    def _draw_enhanced_phone_detections(self, image, phone_boxes):
        """Draw enhanced phone detection visualizations with TRUE MUID-IITR compound boxes"""
        for detection in phone_boxes:
            detection_type = detection['type']
            
            if detection_type == 'COMPOUND_USAGE':
                # TRUE MUID-IITR: Draw compound bounding box
                compound_bbox = detection['compound_bbox']
                hand_bbox = detection['hand_bbox']
                phone_bbox = detection['phone_bbox']
                usage_confidence = detection['usage_confidence']
                
                # Draw compound box (main MUID-IITR feature)
                cv2.rectangle(image, (compound_bbox[0], compound_bbox[1]), 
                             (compound_bbox[2], compound_bbox[3]), (0, 255, 255), 4)  # Yellow
                
                # Draw individual hand and phone boxes inside compound
                cv2.rectangle(image, (hand_bbox[0], hand_bbox[1]), 
                             (hand_bbox[2], hand_bbox[3]), (0, 255, 0), 2)  # Green hand
                cv2.rectangle(image, (phone_bbox[0], phone_bbox[1]), 
                             (phone_bbox[2], phone_bbox[3]), (255, 0, 255), 2)  # Magenta phone
                
                # Draw connection line
                cv2.line(image, detection['hand_center'], detection['phone_center'], 
                        (255, 255, 0), 2)
                
                # Usage confidence label
                label = f"📱USAGE: {usage_confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                
                cv2.rectangle(image, (compound_bbox[0], compound_bbox[1] - label_size[1] - 12), 
                             (compound_bbox[0] + label_size[0] + 10, compound_bbox[1] - 2), 
                             (0, 255, 255), -1)
                cv2.putText(image, label, (compound_bbox[0] + 5, compound_bbox[1] - 8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
                # Distance indicator
                distance_text = f"d:{detection['distance']:.0f}px"
                cv2.putText(image, distance_text, 
                           (compound_bbox[0], compound_bbox[3] + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
            else:
                # Standard detection (COCO, Hand-Fusion, etc.)
                x1, y1, x2, y2 = detection['bbox']
                confidence = detection.get('confidence', 0.0)
                
                # Different colors for different detection methods
                if detection_type == 'COCO':
                    color = self.PHONE_COLOR  # Magenta
                    label_prefix = "📱PHONE"
                elif detection_type == 'Hand-Fusion':
                    color = (0, 255, 255)  # Yellow
                    label_prefix = "📱HAND+"
                else:
                    color = (255, 150, 0)  # Orange
                    label_prefix = "📱OTHER"
                
                # Draw glowing bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)
                cv2.rectangle(image, (x1-2, y1-2), (x2+2, y2+2), color, 1)  # Outer glow
                
                # Enhanced label
                label = f"{label_prefix} {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Label background
                cv2.rectangle(image, (x1, y1 - label_size[1] - 12), 
                             (x1 + label_size[0] + 10, y1 - 2), color, -1)
                cv2.putText(image, label, (x1 + 5, y1 - 8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def process_frame(self, image):
        """Process frame with all detection types"""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with all models
        face_results = self.face_mesh.process(rgb_image)
        hand_results = self.hands.process(rgb_image)
        pose_results = self.pose.process(rgb_image)
        # Enhanced phone detection with hand fusion
        hand_landmarks_list = hand_results.multi_hand_landmarks if hand_results.multi_hand_landmarks else None
        phones_detected = self.detect_phones_enhanced(image, hand_landmarks_list)
        
        # Count detections
        faces_detected = len(face_results.multi_face_landmarks) if face_results.multi_face_landmarks else 0
        hands_detected = len(hand_results.multi_hand_landmarks) if hand_results.multi_hand_landmarks else 0
        pose_detected = 1 if pose_results.pose_landmarks else 0
        
        # Increment animation frame for effects
        self.animation_frame += 1
        
        # Draw face mesh with cool eyes
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                # Draw very subtle face mesh so eyes stand out
                self.mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(60, 60, 60), thickness=1)
                )
                
                # Draw cool but natural eyes
                self.draw_awesome_eyes(image, face_landmarks, self.LEFT_EYE, self.LEFT_IRIS)
                self.draw_awesome_eyes(image, face_landmarks, self.RIGHT_EYE, self.RIGHT_IRIS)
        
        # Draw awesome glowing hands
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.draw_awesome_hands(image, hand_landmarks)
        
        # Draw awesome glowing pose skeleton
        if pose_results.pose_landmarks:
            # Draw awesome skeleton with glow effects
            self.draw_awesome_skeleton(image, pose_results.pose_landmarks)
            
            # Still highlight shoulders with extra glow
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
        
        print("🎯 AWESOME MediaPipe Detection Started!")
        print("=" * 50)
        print("✨ ENHANCED Visual Features:")
        print("   👁️  Eyes: Natural contours + subtle animated iris") 
        print("   🤲 Hands: NEON skeleton with pulsing fingertips")
        print("   💪 Shoulders: HIGHLIGHTED with glow effects")
        print("   🏃 Body: ELECTRIC skeleton with animated joints")
        print("   📱 Phones: TRUE MUID-IITR compound detection + COCO + Hand-Fusion)")
        print("   🔬 MUID-IITR: Compound bounding boxes for actual usage detection")
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
                    cv2.putText(processed_frame, "Phones: Magenta/Orange/Yellow", (250, legend_y + 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                    cv2.putText(processed_frame, "(COCO/Usage/Hand-Fusion)", (250, legend_y + 45), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                # Pause indicator
                if paused:
                    h, w = processed_frame.shape[:2]
                    cv2.rectangle(processed_frame, (w//2 - 100, h//2 - 30), (w//2 + 100, h//2 + 30), (0, 0, 0), -1)
                    cv2.putText(processed_frame, "PAUSED", (w//2 - 80, h//2 + 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                
                cv2.imshow('✨ AWESOME Pose Detection with Glow Effects ✨', processed_frame)
                
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