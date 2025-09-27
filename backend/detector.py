"""
StraightUp - Enhanced Detection System with Noise Monitoring
Real-time detection with MediaPipe, YOLO11n, and Audio Noise Detection

Features:
- All existing visual detection features
- Real-time audio noise monitoring
- Noise-aware pose detection alerts
- Focus/distraction analysis
- Environmental quality assessment
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from ultralytics import YOLO
from collections import deque
import threading
import warnings
warnings.filterwarnings("ignore")

# Import our noise detector (will handle missing pyaudio gracefully)
try:
    from noise_detector import NoiseDetector
    NOISE_DETECTION_AVAILABLE = True
    print("✅ Noise detection module imported successfully")
except ImportError as e:
    print(f"⚠️  Could not import noise detector: {e}")
    print("💡 Run 'uv run python install_pyaudio.py' to install PyAudio")
    NOISE_DETECTION_AVAILABLE = False
    NoiseDetector = None
except Exception as e:
    print(f"⚠️  Noise detector import error: {e}")
    NOISE_DETECTION_AVAILABLE = False
    NoiseDetector = None


class EnhancedPoseDetector:
    """Enhanced pose detection with noise monitoring capabilities"""
    
    def __init__(self, enable_noise_detection=True):
        # Initialize MediaPipe solutions (same as before)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize enhanced phone detection system
        print("Loading enhanced phone detection models...")
        self.yolo_model = YOLO('yolo11n.pt')  # Base COCO model
        print("✅ YOLO COCO model loaded (class 67: cell phone)")
        
        # Phone detection configuration with smoothing
        self.phone_confidence_threshold = 0.15
        self.phone_classes = [67]
        self.phone_detection_history = []
        self.max_history = 10
        
        # Phone detection smoothing to prevent flickering
        self.phone_smoothing_window = 5  # frames to consider
        self.phone_detection_threshold = 0.6  # 60% of frames must have phone
        self.min_session_duration = 1.0  # minimum 1 second before starting session
        self.session_end_delay = 2.0  # wait 2 seconds after last detection before ending
        self.last_stable_phone_time = 0.0
        self.session_candidate_start = None
        
        # Enhanced phone tracking with smooth interpolation
        self._phone_tracks = []
        self._next_track_id = 1
        self.phone_smooth_alpha = 0.6
        self.phone_max_misses = 6
        
        # EMA smoothing for neck center line
        self._neck_base_smooth = None
        self._head_pt_smooth = None
        
        # Initialize noise detection
        self.noise_detector = None
        self.noise_enabled = False
        
        if enable_noise_detection and NOISE_DETECTION_AVAILABLE:
            try:
                print("🔧 Initializing noise detector...")
                self.noise_detector = NoiseDetector(sample_rate=44100, chunk_size=1024)
                self.noise_enabled = False  # Start disabled, user can enable with 'n'
                print("🔊 Noise detection available! Press 'n' to enable")
                print(f"🔍 Noise detector object: {type(self.noise_detector)}")
            except Exception as e:
                print(f"❌ Noise detection failed to initialize: {e}")
                print(f"🔍 Exception type: {type(e)}")
                print("💡 To fix: Install PyAudio with 'pip install pyaudio'")
                self.noise_detector = None
                self.noise_enabled = False
        elif enable_noise_detection and not NOISE_DETECTION_AVAILABLE:
            print("⚠️  PyAudio not available - noise detection disabled")
            print("💡 To enable noise detection:")
            print("   Windows: pip install pyaudio")
            print("   Mac: brew install portaudio && pip install pyaudio") 
            print("   Linux: sudo apt-get install portaudio19-dev && pip install pyaudio")
        else:
            print("🔇 Noise detection disabled by user")
        
        # Enhanced alert system
        self.alerts = deque(maxlen=10)
        self.focus_score = 1.0  # 0.0 = very distracted, 1.0 = very focused
        self.distraction_factors = {
            'noise': 0.0,
            'phone_usage': 0.0,
            'posture': 0.0,
            'movement': 0.0
        }
        
        # Phone alert system
        self.phone_alert_config = {
            'brief_usage_threshold': 5,      # 5 seconds of phone usage
            'extended_usage_threshold': 15,  # 15 seconds for extended usage warning
            'excessive_usage_threshold': 30, # 30 seconds for excessive usage alert
            'break_suggestion_threshold': 60, # 1 minute for break suggestion
            'cooldown_period': 10            # 10 seconds between repeated alerts
        }
        
        self.phone_usage_tracker = {
            'continuous_usage_time': 0.0,
            'total_usage_today': 0.0,
            'last_phone_detected_time': 0.0,
            'last_alert_time': 0.0,
            'usage_sessions': deque(maxlen=20),  # Track recent usage sessions
            'current_session_start': None,
            'breaks_taken': 0,
            'productivity_score': 1.0
        }
        
        # Initialize MediaPipe models (same as before)
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
        
        # MediaPipe landmark indices (same as before)
        self.LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        
        # Enhanced color scheme (same as before)
        self.EYE_COLOR = (200, 200, 255)
        self.IRIS_COLOR = (50, 150, 255)
        self.HAND_COLOR = (0, 255, 150)
        self.SHOULDER_COLOR = (0, 255, 255)
        self.POSE_COLOR = (255, 100, 0)
        self.PHONE_COLOR = (255, 0, 255)
        
        # Neon glow colors
        self.GLOW_EYE = (255, 255, 100)
        self.GLOW_HAND = (100, 255, 200)
        self.GLOW_POSE = (255, 150, 50)
        self.GLOW_SHOULDER = (100, 255, 255)
        
        # Alert colors
        self.ALERT_INFO = (255, 255, 0)     # Yellow
        self.ALERT_WARNING = (0, 165, 255)  # Orange
        self.ALERT_CRITICAL = (0, 0, 255)   # Red
        
        # Animation and trail variables
        self.prev_landmarks = None
        self.trail_points = []
        self.animation_frame = 0
        self.pulse_factor = 0
    
    def start_noise_detection(self):
        """Start noise detection if available"""
        if not self.noise_detector:
            print("⚠️  No noise detector available")
            return False
            
        if self.noise_enabled:
            print("ℹ️  Noise detection already running")
            return True
            
        try:
            self.noise_detector.start_detection()
            self.noise_enabled = True
            print("🎤 Noise detection started successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to start noise detection: {e}")
            self.noise_enabled = False
            return False
    
    def stop_noise_detection(self):
        """Stop noise detection"""
        if not self.noise_detector:
            print("⚠️  No noise detector available")
            return
            
        if not self.noise_enabled:
            print("ℹ️  Noise detection already stopped")
            return
            
        try:
            self.noise_detector.stop_detection()
            self.noise_enabled = False
            print("🔇 Noise detection stopped successfully")
        except Exception as e:
            print(f"⚠️  Error stopping noise detection: {e}")
            self.noise_enabled = False
    
    def analyze_environment(self):
        """Analyze environmental factors affecting focus"""
        # Reset noise distraction if noise detection is off
        if not self.noise_enabled or not self.noise_detector:
            self.distraction_factors['noise'] = 0.0
            self.calculate_focus_score()
            return
        
        # Get noise information
        noise_info = self.noise_detector.get_noise_info()
        
        # Calculate noise distraction factor (0.0 = no distraction, 1.0 = max distraction)
        if noise_info['is_noisy']:
            self.distraction_factors['noise'] = min(noise_info['noise_level'] / 0.1, 1.0)
        else:
            self.distraction_factors['noise'] = 0.0
        
        # Check for noise-related alerts
        noise_alerts = self.noise_detector.get_noise_alerts()
        for alert in noise_alerts:
            self.add_alert(alert['type'], f"NOISE: {alert['message']}", alert.get('recommendation', ''))
        
        # Calculate overall focus score
        self.calculate_focus_score()
    
    def calculate_focus_score(self):
        """Calculate overall focus score based on distraction factors"""
        # Weighted average of distraction factors
        weights = {
            'noise': 0.3,
            'phone_usage': 0.4,
            'posture': 0.2,
            'movement': 0.1
        }
        
        total_distraction = sum(
            self.distraction_factors[factor] * weights[factor]
            for factor in weights
        )
        
        # Focus score is inverse of distraction (1.0 = perfect focus, 0.0 = completely distracted)
        self.focus_score = max(0.0, 1.0 - total_distraction)
    
    def add_alert(self, alert_type, message, recommendation=""):
        """Add an alert to the alert queue"""
        self.alerts.append({
            'type': alert_type,
            'message': message,
            'recommendation': recommendation,
            'timestamp': time.time()
        })
    
    def get_motivational_phone_message(self, session_type, duration):
        """Get contextual motivational messages for phone usage"""
        messages = {
            'brief': [
                "Great! Quick and focused phone check ✅",
                "Perfect timing! Brief check completed 🎯",
                "Excellent focus! Short phone interaction 💪"
            ],
            'moderate': [
                "Good control! Consider wrapping up soon ⏰",
                "You're doing well! Try to finish up 👍",
                "Nice balance! Maybe time to refocus? 🎯"
            ],
            'extended': [
                "Phone break getting long - time to refocus? 📱➡️💼",
                "Extended usage detected - consider a real break instead? ☕",
                "Long phone session - your focus is waiting! 🧠"
            ],
            'excessive': [
                "Time for a real break? Step away from screens! 🚶‍♂️",
                "Excessive screen time - try a physical activity? 🏃‍♂️",
                "Take care of yourself - maybe some fresh air? 🌿"
            ]
        }
        
        import random
        return random.choice(messages.get(session_type, messages['moderate']))
    
    def get_focus_tips(self):
        """Get random focus improvement tips"""
        tips = [
            "💡 Try the 20-20-20 rule: Every 20 min, look at something 20 feet away for 20 seconds",
            "🧘 Take 3 deep breaths to recenter your focus",
            "🎵 Consider background music or white noise for concentration",
            "☕ Stay hydrated! Dehydration affects concentration",
            "📝 Write down distracting thoughts to address later",
            "🎯 Set a specific goal for this work session",
            "⏰ Use the Pomodoro technique: 25 min work, 5 min break",
            "🌱 Add a plant to your workspace for better air quality",
            "📱 Put your phone in another room for deep work",
            "✨ Clean workspace = clean mind. Tidy up!"
        ]
        
        import random
        return random.choice(tips)
    
    def analyze_phone_usage(self, phones_detected):
        """Enhanced phone usage analysis with smoothed detection to prevent spam alerts"""
        current_time = time.time()
        
        # Calculate smoothed phone detection over recent history
        if len(self.phone_detection_history) >= self.phone_smoothing_window:
            recent_detections = self.phone_detection_history[-self.phone_smoothing_window:]
            phone_detection_ratio = sum(1 for x in recent_detections if x > 0) / len(recent_detections)
            stable_phone_detected = phone_detection_ratio >= self.phone_detection_threshold
        else:
            stable_phone_detected = phones_detected > 0
        
        if stable_phone_detected:
            self.last_stable_phone_time = current_time
            
            # Check if we should start a new session
            if self.phone_usage_tracker['current_session_start'] is None:
                if self.session_candidate_start is None:
                    # Start candidate session
                    self.session_candidate_start = current_time
                elif current_time - self.session_candidate_start >= self.min_session_duration:
                    # Candidate session has lasted long enough, make it official
                    self.phone_usage_tracker['current_session_start'] = self.session_candidate_start
                    self.session_candidate_start = None
                    print("📱 Phone usage session started (confirmed)")
            
            # Update continuous usage tracking if session is active
            if self.phone_usage_tracker['current_session_start'] is not None:
                self.phone_usage_tracker['continuous_usage_time'] = (
                    current_time - self.phone_usage_tracker['current_session_start']
                )
                self.phone_usage_tracker['last_phone_detected_time'] = current_time
                
                # Calculate distraction factor based on usage duration
                usage_time = self.phone_usage_tracker['continuous_usage_time']
                self.distraction_factors['phone_usage'] = min(usage_time / 30.0, 1.0)  # Max at 30 seconds
                
                # Generate smart phone alerts
                self._generate_phone_alerts(current_time, usage_time)
            
        else:
            # Phone not stably detected
            # Cancel candidate session if it hasn't been confirmed yet
            if (self.session_candidate_start is not None and 
                current_time - self.last_stable_phone_time > self.session_end_delay):
                self.session_candidate_start = None
            
            # End confirmed session only after delay period
            if (self.phone_usage_tracker['current_session_start'] is not None and 
                current_time - self.last_stable_phone_time > self.session_end_delay):
                
                session_duration = self.last_stable_phone_time - self.phone_usage_tracker['current_session_start']
                
                # Only record sessions that lasted a reasonable time
                if session_duration >= self.min_session_duration:
                    self.phone_usage_tracker['usage_sessions'].append({
                        'duration': session_duration,
                        'timestamp': current_time,
                        'type': self._classify_usage_session(session_duration)
                    })
                    
                    self.phone_usage_tracker['total_usage_today'] += session_duration
                    print(f"📱 Phone usage session ended: {session_duration:.1f}s")
                    
                    # Positive reinforcement for short usage with motivational messages
                    if session_duration < self.phone_alert_config['brief_usage_threshold']:
                        motivational_msg = self.get_motivational_phone_message('brief', session_duration)
                        self.add_alert('info', motivational_msg, 'Keep up the great focus patterns!')
                
                self.phone_usage_tracker['current_session_start'] = None
                self.phone_usage_tracker['continuous_usage_time'] = 0.0
            
            # Gradually reduce distraction factor when phone is not in use
            if current_time - self.last_stable_phone_time > self.session_end_delay:
                self.distraction_factors['phone_usage'] = max(0.0, self.distraction_factors['phone_usage'] - 0.05)
    
    def _classify_usage_session(self, duration):
        """Classify phone usage session type"""
        if duration < self.phone_alert_config['brief_usage_threshold']:
            return 'brief'
        elif duration < self.phone_alert_config['extended_usage_threshold']:
            return 'moderate'
        elif duration < self.phone_alert_config['excessive_usage_threshold']:
            return 'extended'
        else:
            return 'excessive'
    
    def _generate_phone_alerts(self, current_time, usage_time):
        """Generate contextual phone usage alerts"""
        config = self.phone_alert_config
        tracker = self.phone_usage_tracker
        
        # Check cooldown period
        time_since_last_alert = current_time - tracker['last_alert_time']
        if time_since_last_alert < config['cooldown_period']:
            return
        
        # Brief usage reminder (gentle) with motivational messages
        if (usage_time >= config['brief_usage_threshold'] and 
            usage_time < config['extended_usage_threshold']):
            motivational_msg = self.get_motivational_phone_message('moderate', usage_time)
            self.add_alert('info', motivational_msg, self.get_focus_tips())
            tracker['last_alert_time'] = current_time
        
        # Extended usage warning with engaging messages
        elif (usage_time >= config['extended_usage_threshold'] and 
              usage_time < config['excessive_usage_threshold']):
            motivational_msg = self.get_motivational_phone_message('extended', usage_time)
            self.add_alert('warning', motivational_msg, 'Your productivity is waiting for you!')
            tracker['last_alert_time'] = current_time
        
        # Excessive usage alert with supportive guidance
        elif (usage_time >= config['excessive_usage_threshold'] and 
              usage_time < config['break_suggestion_threshold']):
            motivational_msg = self.get_motivational_phone_message('excessive', usage_time)
            self.add_alert('critical', motivational_msg, 'Consider a mindful break instead')
            tracker['last_alert_time'] = current_time
        
        # Break suggestion for very long usage with wellness focus
        elif usage_time >= config['break_suggestion_threshold']:
            wellness_msg = self.get_motivational_phone_message('excessive', usage_time)
            self.add_alert('critical', wellness_msg, 'Physical activity > screen scrolling!')
            tracker['last_alert_time'] = current_time
    
    def get_phone_usage_stats(self):
        """Get comprehensive phone usage statistics"""
        tracker = self.phone_usage_tracker
        current_time = time.time()
        
        # Calculate productivity score based on recent usage patterns
        recent_sessions = [s for s in tracker['usage_sessions'] 
                          if current_time - s['timestamp'] < 300]  # Last 5 minutes
        
        if recent_sessions:
            total_recent_usage = sum(s['duration'] for s in recent_sessions)
            productivity_impact = min(total_recent_usage / 60.0, 1.0)  # Impact over 1 minute
            tracker['productivity_score'] = max(0.0, 1.0 - productivity_impact)
        else:
            tracker['productivity_score'] = min(1.0, tracker['productivity_score'] + 0.1)
        
        return {
            'current_session_duration': tracker['continuous_usage_time'],
            'total_usage_today': tracker['total_usage_today'],
            'productivity_score': tracker['productivity_score'],
            'recent_sessions': len(recent_sessions),
            'breaks_taken': tracker['breaks_taken'],
            'session_type': self._classify_usage_session(tracker['continuous_usage_time']),
            'is_in_session': tracker['current_session_start'] is not None
        }
    
    def analyze_posture(self, pose_detected, landmarks=None):
        """Analyze posture for focus assessment"""
        if not pose_detected or landmarks is None:
            self.distraction_factors['posture'] = 0.3  # Some penalty for no pose
            return
        
        # Simple posture analysis - check shoulder alignment
        try:
            left_shoulder = landmarks.landmark[11]
            right_shoulder = landmarks.landmark[12]
            
            # Calculate shoulder angle
            shoulder_angle = abs(left_shoulder.y - right_shoulder.y)
            
            # Good posture has aligned shoulders (small angle)
            if shoulder_angle > 0.05:  # Threshold for poor posture
                self.distraction_factors['posture'] = min(shoulder_angle * 10, 1.0)
                if shoulder_angle > 0.1:
                    self.add_alert('info', 'Posture check: Uneven shoulders detected',
                                 'Try to align your shoulders for better posture')
            else:
                self.distraction_factors['posture'] = 0.0
                
        except Exception:
            self.distraction_factors['posture'] = 0.0
    
    # -------------------- Utility Functions (same as before) --------------------
    def _ema(self, prev, cur, a=0.3):
        """Exponential Moving Average for smooth positioning"""
        if prev is None:
            return cur
        return (int(a*cur[0] + (1-a)*prev[0]), int(a*cur[1] + (1-a)*prev[1]))
    
    def draw_shoulder_highlight(self, image, landmarks):
        """Highlight shoulders specifically (same as before)"""
        h, w = image.shape[:2]
        
        left_shoulder = landmarks.landmark[11]
        right_shoulder = landmarks.landmark[12]
        
        left_x, left_y = int(left_shoulder.x * w), int(left_shoulder.y * h)
        right_x, right_y = int(right_shoulder.x * w), int(right_shoulder.y * h)
        
        cv2.circle(image, (left_x, left_y), 10, self.SHOULDER_COLOR, -1)
        cv2.circle(image, (right_x, right_y), 10, self.SHOULDER_COLOR, -1)
        cv2.line(image, (left_x, left_y), (right_x, right_y), self.SHOULDER_COLOR, 3)
        
        cv2.putText(image, "L.SHOULDER", (left_x - 60, left_y - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.SHOULDER_COLOR, 1)
        cv2.putText(image, "R.SHOULDER", (right_x + 10, right_y - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.SHOULDER_COLOR, 1)
    
    def draw_glow_effect(self, image, points, color, glow_color, thickness=3):
        """Draw glowing lines with neon effect (same as before)"""
        if len(points) < 2:
            return
        
        for i in range(len(points) - 1):
            cv2.line(image, points[i], points[i + 1], glow_color, thickness + 6)
        
        for i in range(len(points) - 1):
            cv2.line(image, points[i], points[i + 1], glow_color, thickness + 3)
        
        for i in range(len(points) - 1):
            cv2.line(image, points[i], points[i + 1], color, thickness)
    
    def draw_neon_circle(self, image, center, radius, color, glow_color):
        """Draw a neon glowing circle (same as before)"""
        cv2.circle(image, center, radius + 6, glow_color, -1)
        cv2.circle(image, center, radius + 3, glow_color, -1)
        cv2.circle(image, center, radius, color, -1)
        cv2.circle(image, center, max(1, radius // 2), (255, 255, 255), -1)
    
    def draw_awesome_skeleton(self, image, landmarks):
        """Enhanced glowing skeleton with neck center line mapping (same as before)"""
        h, w = image.shape[:2]
        
        points = []
        for landmark in landmarks.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            points.append((x, y))
        
        # Torso connections
        torso_connections = [(11, 12), (11, 23), (12, 24), (23, 24)]
        for start_idx, end_idx in torso_connections:
            if start_idx < len(points) and end_idx < len(points):
                self.draw_glow_effect(image, [points[start_idx], points[end_idx]], 
                                    self.POSE_COLOR, self.GLOW_POSE, 4)
        
        # Arms
        arm_connections = [(11, 13), (13, 15), (12, 14), (14, 16)]
        for start_idx, end_idx in arm_connections:
            if start_idx < len(points) and end_idx < len(points):
                self.draw_glow_effect(image, [points[start_idx], points[end_idx]], 
                                    self.HAND_COLOR, self.GLOW_HAND, 3)
        
        # Legs
        leg_connections = [(23, 25), (25, 27), (24, 26), (26, 28)]
        leg_color = (255, 100, 255)
        glow_leg = (255, 150, 255)
        for start_idx, end_idx in leg_connections:
            if start_idx < len(points) and end_idx < len(points):
                self.draw_glow_effect(image, [points[start_idx], points[end_idx]], 
                                    leg_color, glow_leg, 3)
        
        # NECK LINES AND CENTER MAPPING
        if len(points) > 12:
            self.draw_glow_effect(image, [points[11], points[0]],
                                  self.SHOULDER_COLOR, self.GLOW_SHOULDER, 3)
            self.draw_glow_effect(image, [points[12], points[0]],
                                  self.SHOULDER_COLOR, self.GLOW_SHOULDER, 3)

            mx = int((points[11][0] + points[12][0]) / 2)
            my = int((points[11][1] + points[12][1]) / 2)
            neck_base_raw = (mx, my)
            head_pt_raw = points[0]

            neck_base = self._ema(self._neck_base_smooth, neck_base_raw, a=0.7) if self._neck_base_smooth else neck_base_raw
            head_pt   = self._ema(self._head_pt_smooth,   head_pt_raw,   a=0.7) if self._head_pt_smooth   else head_pt_raw
            self._neck_base_smooth = neck_base
            self._head_pt_smooth   = head_pt

            cv2.line(image, neck_base, head_pt, self.SHOULDER_COLOR, 2, lineType=cv2.LINE_AA)
            self.draw_neon_circle(image, neck_base, 8, self.SHOULDER_COLOR, self.GLOW_SHOULDER)
            cv2.putText(image, "NECK", (neck_base[0] - 20, neck_base[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.SHOULDER_COLOR, 1)
        
        # Draw glowing joint points
        joint_indices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        for idx in joint_indices:
            if idx < len(points):
                pulse = int(10 + 5 * np.sin(self.animation_frame * 0.2 + idx))
                joint_color = self.POSE_COLOR
                if idx in [11, 12]:
                    joint_color = self.SHOULDER_COLOR
                    self.draw_neon_circle(image, points[idx], pulse, joint_color, self.GLOW_SHOULDER)
                else:
                    self.draw_neon_circle(image, points[idx], pulse, joint_color, self.GLOW_POSE)
    
    def draw_awesome_hands(self, image, hand_landmarks):
        """Draw awesome glowing hands with particle effects (same as before)"""
        h, w = image.shape[:2]
        
        hand_points = []
        for landmark in hand_landmarks.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            hand_points.append((x, y))
        
        connections = [
            [0, 1, 2, 3, 4],
            [0, 5, 6, 7, 8],
            [0, 9, 10, 11, 12],
            [0, 13, 14, 15, 16],
            [0, 17, 18, 19, 20]
        ]
        
        for finger in connections:
            finger_points = [hand_points[i] for i in finger if i < len(hand_points)]
            if len(finger_points) > 1:
                self.draw_glow_effect(image, finger_points, self.HAND_COLOR, self.GLOW_HAND, 2)
        
        fingertips = [4, 8, 12, 16, 20]
        for tip_idx in fingertips:
            if tip_idx < len(hand_points):
                pulse = int(8 + 3 * np.sin(self.animation_frame * 0.3 + tip_idx))
                self.draw_neon_circle(image, hand_points[tip_idx], pulse, 
                                    self.HAND_COLOR, self.GLOW_HAND)
    
    def draw_awesome_eyes(self, image, landmarks, eye_indices, iris_indices):
        """Draw cool but natural-looking eyes with subtle effects (same as before)"""
        h, w = image.shape[:2]
        
        eye_points = []
        for idx in eye_indices:
            if idx < len(landmarks.landmark):
                x = int(landmarks.landmark[idx].x * w)
                y = int(landmarks.landmark[idx].y * h)
                eye_points.append((x, y))
                cv2.circle(image, (x, y), 1, self.EYE_COLOR, -1)
        
        if len(eye_points) > 3:
            eye_poly = np.array(eye_points, dtype=np.int32)
            cv2.polylines(image, [eye_poly], True, (150, 200, 255), 2)
            cv2.polylines(image, [eye_poly], True, self.EYE_COLOR, 1)
        
        iris_points = []
        for idx in iris_indices:
            if idx < len(landmarks.landmark):
                x = int(landmarks.landmark[idx].x * w)
                y = int(landmarks.landmark[idx].y * h)
                iris_points.append((x, y))
        
        if len(iris_points) >= 4:
            iris_center = np.mean(iris_points, axis=0).astype(int)
            base_radius = 8
            pulse = int(base_radius + 2 * np.sin(self.animation_frame * 0.1))
            
            cv2.circle(image, tuple(iris_center), pulse + 3, (100, 100, 255), -1)
            cv2.circle(image, tuple(iris_center), pulse, self.IRIS_COLOR, -1)
            cv2.circle(image, tuple(iris_center), max(1, pulse // 3), (255, 255, 255), -1)
    
    # -------------------- Enhanced Phone Detection (same as before) --------------------
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
                self._phone_tracks.append({
                    "id": self._next_track_id,
                    "bbox": det['bbox'],
                    "type": det['type'],
                    "conf": det['confidence'],
                    "misses": 0
                })
                self._next_track_id += 1

        kept = []
        for idx, t in enumerate(self._phone_tracks):
            if idx in unmatched_tracks:
                t['misses'] += 1
            if t['misses'] <= self.phone_max_misses:
                kept.append(t)
        self._phone_tracks = kept

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
        """Enhanced phone detection with smooth tracking"""
        hand_lms_list = hand_landmarks if hand_landmarks else None
        raw_phone_dets = self._detect_phones_all_sources(image, hand_lms_list)
        smoothed_boxes = self._update_phone_tracks_and_get_boxes(raw_phone_dets)
        self._draw_enhanced_phone_detections(image, smoothed_boxes)
        phones_detected_now = len([t for t in self._phone_tracks if t['misses'] == 0])
        
        self.phone_detection_history.append(phones_detected_now)
        if len(self.phone_detection_history) > self.max_history:
            self.phone_detection_history.pop(0)
        
        return phones_detected_now
    
    def draw_enhanced_info_panel(self, image, faces, hands, pose, phones):
        """Draw enhanced info panel with noise and focus information"""
        h, w = image.shape[:2]
        
        # Main status box (made wider for phone stats)
        cv2.rectangle(image, (10, 10), (550, 220), (0, 0, 0), -1)
        cv2.rectangle(image, (10, 10), (550, 220), (255, 255, 255), 2)
        
        # Basic detection info  
        cv2.putText(image, f"Faces: {faces}", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(image, f"Hands: {hands}", (20, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(image, f"Pose: {'Yes' if pose else 'No'}", (20, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(image, f"Phones: {phones}", (20, 95), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Focus score
        focus_color = self.ALERT_INFO
        if self.focus_score > 0.7:
            focus_color = (0, 255, 0)  # Green for good focus
        elif self.focus_score < 0.4:
            focus_color = self.ALERT_CRITICAL  # Red for poor focus
        
        cv2.putText(image, f"Focus: {self.focus_score:.2f}", (20, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, focus_color, 2)
        
        # Phone detection smoothing status
        if len(self.phone_detection_history) >= self.phone_smoothing_window:
            recent_detections = self.phone_detection_history[-self.phone_smoothing_window:]
            detection_ratio = sum(1 for x in recent_detections if x > 0) / len(recent_detections)
            smooth_color = (0, 255, 0) if detection_ratio >= self.phone_detection_threshold else (100, 100, 100)
            cv2.putText(image, f"Phone Smooth: {detection_ratio:.1f}", (280, 95), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, smooth_color, 2)
        
        # Session status
        session_status = "None"
        if self.phone_usage_tracker['current_session_start'] is not None:
            session_status = "Active"
        elif self.session_candidate_start is not None:
            session_status = "Candidate"
        
        cv2.putText(image, f"Session: {session_status}", (280, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Phone usage statistics
        phone_stats = self.get_phone_usage_stats()
        
        # Current session info
        if phone_stats['is_in_session']:
            session_color = self.ALERT_WARNING if phone_stats['current_session_duration'] > 15 else self.ALERT_INFO
            cv2.putText(image, f"Phone Session: {phone_stats['current_session_duration']:.1f}s", 
                       (280, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, session_color, 1)
            cv2.putText(image, f"Type: {phone_stats['session_type'].upper()}", 
                       (280, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, session_color, 1)
        else:
            cv2.putText(image, "Phone: Not in use", (280, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Productivity score
        prod_color = (0, 255, 0) if phone_stats['productivity_score'] > 0.7 else (0, 165, 255) if phone_stats['productivity_score'] > 0.4 else (0, 0, 255)
        cv2.putText(image, f"Productivity: {phone_stats['productivity_score']:.2f}", 
                   (280, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, prod_color, 1)
        
        # Daily usage summary
        cv2.putText(image, f"Total Today: {phone_stats['total_usage_today']:.0f}s", 
                   (280, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(image, f"Recent Sessions: {phone_stats['recent_sessions']}", 
                   (280, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Noise information with better status display
        if self.noise_detector and self.noise_enabled:
            try:
                noise_info = self.noise_detector.get_noise_info()
                noise_color = (0, 255, 0) if not noise_info['is_noisy'] else (0, 0, 255)
                cv2.putText(image, f"Noise: {noise_info['category']}", (20, 140), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, noise_color, 2)
                cv2.putText(image, f"Level: {noise_info['noise_level']:.3f}", (20, 160), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, noise_color, 1)
            except Exception as e:
                cv2.putText(image, "Noise: ERROR", (20, 140), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        elif self.noise_detector and not self.noise_enabled:
            cv2.putText(image, "Noise: OFF (Press 'n' to enable)", (20, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
        else:
            cv2.putText(image, "Noise: NOT AVAILABLE", (20, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
        
        # Distraction factors
        y_offset = 180
        for factor, value in self.distraction_factors.items():
            if value > 0.1:  # Only show significant distractions
                factor_color = self.ALERT_WARNING if value > 0.5 else self.ALERT_INFO
                cv2.putText(image, f"{factor}: {value:.2f}", (20, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, factor_color, 1)
                y_offset += 15
        
        # Phone usage trend indicator
        if phone_stats['recent_sessions'] > 0:
            trend_x = 450
            trend_y = 140
            trend_color = (0, 255, 0) if phone_stats['productivity_score'] > 0.7 else (0, 0, 255)
            
            # Draw small trend indicator
            cv2.circle(image, (trend_x, trend_y), 8, trend_color, -1)
            cv2.putText(image, "TREND", (trend_x - 20, trend_y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, trend_color, 1)
    
    def draw_alerts(self, image):
        """Draw recent alerts on the image"""
        if not self.alerts:
            return
        
        h, w = image.shape[:2]
        y_start = h - 150
        
        # Show last 5 alerts
        recent_alerts = list(self.alerts)[-5:]
        
        for i, alert in enumerate(recent_alerts):
            y_pos = y_start + i * 30
            
            # Choose color based on alert type
            if alert['type'] == 'critical':
                color = self.ALERT_CRITICAL
            elif alert['type'] == 'warning':
                color = self.ALERT_WARNING
            else:
                color = self.ALERT_INFO
            
            # Draw alert background
            text_size = cv2.getTextSize(alert['message'][:60], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(image, (10, y_pos - 15), (20 + text_size[0], y_pos + 5), (0, 0, 0), -1)
            cv2.rectangle(image, (10, y_pos - 15), (20 + text_size[0], y_pos + 5), color, 1)
            
            # Draw alert text
            cv2.putText(image, alert['message'][:60], (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def process_frame(self, image):
        """Process one frame with enhanced analysis"""
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

        # Analyze phone usage patterns
        self.analyze_phone_usage(phones_detected)
        
        # Analyze posture
        self.analyze_posture(pose_detected, pose_results.pose_landmarks if pose_detected else None)
        
        # Analyze environmental factors
        self.analyze_environment()

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

        # Draw pose + shoulders + neck line
        if pose_detected:
            self.draw_awesome_skeleton(image, pose_results.pose_landmarks)
            self.draw_shoulder_highlight(image, pose_results.pose_landmarks)

        return image, faces_detected, hands_detected, pose_detected, phones_detected
    
    def run_webcam(self):
        """Run the enhanced detection system on webcam"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("🎯 StraightUp Enhanced Detection System Started!")
        print("=" * 60)
        print("✨ Visual Features:")
        print("   👁️  Eyes: Natural contours + animated iris")
        print("   🤲 Hands: NEON skeleton with pulsing fingertips")
        print("   💪 Shoulders: HIGHLIGHTED with glow effects")
        print("   🏃 Body: ELECTRIC skeleton with animated joints")
        print("   📍 Neck: Smooth center line mapping (EMA smoothed)")
        print("   📱 Phones: Enhanced YOLO11n tracking with smoothing")
        print("   🔊 Audio: Real-time noise monitoring and analysis")
        print("   🎯 Focus: AI-powered focus and distraction analysis")
        print("   ✨ All with GLOW EFFECTS and smooth animations!")
        print("\n⌨️  Controls:")
        print("   'q' or ESC: Quit application")
        print("   's': Save current frame")
        print("   'i': Toggle info display")
        print("   'n': Toggle noise detection")
        print("   'a': Toggle alert display")
        print("   'p': Reset phone usage statistics")
        print("   'r': Show phone usage report")
        print("   Space: Pause/Resume detection")
        print("=" * 60)
        
        # Don't auto-start noise detection - let user enable it with 'n'
        # if self.noise_detector:
        #     self.start_noise_detection()
        
        # Performance tracking
        prev_time = time.time()
        fps_counter = 0
        fps = 0
        show_info = True
        show_alerts = True
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
                
                # Draw enhanced info panel
                if show_info:
                    cv2.putText(processed_frame, f"FPS: {fps:.1f}", (470, 35), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    self.draw_enhanced_info_panel(processed_frame, faces, hands, pose, phones)
                
                # Draw noise indicator if enabled
                if self.noise_enabled and self.noise_detector:
                    self.noise_detector.draw_noise_indicator(processed_frame, (580, 60))
                    self.noise_detector.draw_noise_history(processed_frame, (580, 120), 200, 80)
                elif self.noise_detector and not self.noise_enabled:
                    # Show "disabled" indicator
                    cv2.rectangle(processed_frame, (580, 60), (780, 100), (50, 50, 50), -1)
                    cv2.rectangle(processed_frame, (580, 60), (780, 100), (128, 128, 128), 2)
                    cv2.putText(processed_frame, "NOISE: DISABLED", (590, 85),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
                    cv2.putText(processed_frame, "Press 'n' to enable", (590, 95),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128, 128, 128), 1)
                
                # Draw phone usage graph
                self.draw_phone_usage_graph(processed_frame, (580, 220), 200, 60)
                
                # Draw alerts
                if show_alerts:
                    self.draw_alerts(processed_frame)
                
                # Pause indicator
                if paused:
                    h, w = processed_frame.shape[:2]
                    cv2.rectangle(processed_frame, (w//2 - 100, h//2 - 30), (w//2 + 100, h//2 + 30), (0, 0, 0), -1)
                    cv2.putText(processed_frame, "PAUSED", (w//2 - 80, h//2 + 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                
                cv2.imshow('StraightUp - Enhanced Detection with Noise Monitoring', processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord('s'):  # Save frame
                    timestamp = int(time.time())
                    filename = f'enhanced_detection_{timestamp}.jpg'
                    cv2.imwrite(filename, processed_frame)
                    print(f"💾 Frame saved as {filename}")
                elif key == ord('i'):  # Toggle info
                    show_info = not show_info
                    print(f"ℹ️  Info display: {'ON' if show_info else 'OFF'}")
                elif key == ord('n'):  # Toggle noise detection
                    print(f"🔍 Debug: noise_detector is None? {self.noise_detector is None}")
                    print(f"🔍 Debug: noise_detector type: {type(self.noise_detector)}")
                    print(f"🔍 Debug: NOISE_DETECTION_AVAILABLE: {NOISE_DETECTION_AVAILABLE}")
                    
                    if self.noise_detector:
                        if self.noise_enabled:
                            self.stop_noise_detection()
                            print("🔇 Noise detection: OFF")
                        else:
                            self.start_noise_detection()
                            print("🔊 Noise detection: ON")
                    else:
                        print("⚠️  Noise detection not available (no microphone or PyAudio not installed)")
                        print("🔍 This should not happen if initialization succeeded!")
                elif key == ord('a'):  # Toggle alerts
                    show_alerts = not show_alerts
                    print(f"🚨 Alert display: {'ON' if show_alerts else 'OFF'}")
                elif key == ord('p'):  # Phone usage reset
                    self.phone_usage_tracker['total_usage_today'] = 0.0
                    self.phone_usage_tracker['usage_sessions'].clear()
                    self.phone_usage_tracker['current_session_start'] = None
                    self.phone_usage_tracker['continuous_usage_time'] = 0.0
                    print("📱 Phone usage stats reset!")
                elif key == ord('r'):  # Show phone usage report
                    self._show_phone_usage_report()
                elif key == ord(' '):  # Space to pause
                    paused = not paused
                    print(f"⏸️  {'Paused' if paused else 'Resumed'}")
        
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user")
        finally:
            if self.noise_detector:
                self.stop_noise_detection()
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Enhanced detection completed")
    
    def _show_phone_usage_report(self):
        """Display comprehensive phone usage report"""
        stats = self.get_phone_usage_stats()
        sessions = list(self.phone_usage_tracker['usage_sessions'])
        
        print("\n" + "="*50)
        print("📱 PHONE USAGE REPORT")
        print("="*50)
        print(f"📊 Current Session: {stats['current_session_duration']:.1f}s ({stats['session_type']})")
        print(f"📈 Productivity Score: {stats['productivity_score']:.2f}")
        print(f"⏱️  Total Usage Today: {stats['total_usage_today']:.0f}s ({stats['total_usage_today']/60:.1f} min)")
        print(f"🔄 Recent Sessions (5min): {stats['recent_sessions']}")
        print(f"☕ Breaks Taken: {stats['breaks_taken']}")
        
        if sessions:
            print(f"\n📋 Recent Session History ({len(sessions)} sessions):")
            for i, session in enumerate(sessions[-5:]):  # Show last 5 sessions
                session_type = session['type'].upper()
                duration = session['duration']
                print(f"   {i+1}. {session_type}: {duration:.1f}s")
                
            # Usage pattern analysis
            brief_sessions = sum(1 for s in sessions if s['type'] == 'brief')
            extended_sessions = sum(1 for s in sessions if s['type'] in ['extended', 'excessive'])
            
            print(f"\n📈 Usage Patterns:")
            print(f"   ✅ Brief checks: {brief_sessions}")
            print(f"   ⚠️  Extended usage: {extended_sessions}")
            
            if brief_sessions > extended_sessions:
                print("   🎉 Good! You're maintaining focused phone usage patterns.")
            else:
                print("   💡 Consider shorter phone interactions for better focus.")
        
        print("="*50)
    
    def draw_phone_usage_graph(self, image, position=(50, 250), width=300, height=80):
        """Draw phone usage history graph"""
        if not self.phone_usage_tracker['usage_sessions']:
            return
        
        x, y = position
        sessions = list(self.phone_usage_tracker['usage_sessions'])[-20:]  # Last 20 sessions
        
        if len(sessions) < 2:
            return
        
        # Background
        cv2.rectangle(image, (x, y), (x + width, y + height), (30, 30, 30), -1)
        cv2.rectangle(image, (x, y), (x + width, y + height), (255, 255, 255), 1)
        
        # Calculate bar width
        bar_width = width // len(sessions)
        max_duration = max(s['duration'] for s in sessions)
        max_duration = max(max_duration, 30)  # At least 30 seconds scale
        
        # Draw session bars
        for i, session in enumerate(sessions):
            bar_x = x + i * bar_width
            bar_height = int((session['duration'] / max_duration) * height)
            bar_y = y + height - bar_height
            
            # Color based on session type
            if session['type'] == 'brief':
                color = (0, 255, 0)  # Green
            elif session['type'] == 'moderate':
                color = (0, 255, 255)  # Yellow
            elif session['type'] == 'extended':
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 0, 255)  # Red
            
            cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width - 1, y + height), color, -1)
        
        # Labels
        cv2.putText(image, "Phone Usage History", (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(image, f"Max: {max_duration:.0f}s", (x + width - 60, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)


def main():
    """Main function to run the enhanced detection system"""
    print("🚀 Starting Enhanced MediaPipe Detection System with Noise Monitoring...")
    detector = EnhancedPoseDetector(enable_noise_detection=True)
    detector.run_webcam()


if __name__ == "__main__":
    main()