"""
StraightUp - Integrated Detection System
Real-time detection with MediaPipe, YOLO11n, Posture Analysis, and System Actions

Combined Features:
- Enhanced pose detection with eye contours and animated iris tracking
- Full 21-point hand skeleton with glow effects  
- Complete body pose with neck center line mapping
- Enhanced phone detection with smooth tracking
- Intelligent posture analysis with hysteresis
- System actions (notifications and screen dimming)
- Real-time webcam processing with comprehensive controls
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import math
import ctypes
from ctypes import wintypes
from ultralytics import YOLO
from collections import deque
from dataclasses import dataclass

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

# ---- Optional toast support (non-fatal if missing) ----
try:
    from win10toast import ToastNotifier   # pip install win10toast
    _toaster = ToastNotifier()
except Exception:
    _toaster = None

# ======================= POSTURE ANALYSIS =======================

# MediaPipe pose landmarks
POSE_NOSE = 0
POSE_L_EYE = 2
POSE_R_EYE = 5
POSE_L_SHO = 11
POSE_R_SHO = 12

def _pt(landmark, w, h):
    return (landmark.x * w, landmark.y * h, landmark.z)  # x,y in px, z in normalized units

def _vec(a, b):
    return (b[0]-a[0], b[1]-a[1])

def _len(v):
    return math.hypot(v[0], v[1])

def _angle_deg(u, v):
    dot = u[0]*v[0] + u[1]*v[1]
    nu = _len(u); nv = _len(v)
    if nu*nv == 0: return 0.0
    cosv = max(-1.0, min(1.0, dot/(nu*nv)))
    return math.degrees(math.acos(cosv))

def _ema(prev, cur, a=0.3):
    return cur if prev is None else (a*cur + (1-a)*prev)

@dataclass
class Thresholds:
    # normalized by shoulder width unless noted
    neck_flex_bad_deg: float = 9.5     # angle from vertical
    neck_flex_warn_deg: float = 5.0
    fhd_bad_ratio: float = 0.35         # forward-head: horizontal nose offset / shoulder width
    fhd_warn_ratio: float = 0.25
    shoulder_slope_bad_deg: float = 180
    shoulder_slope_warn_deg: float = 170
    head_tilt_bad_deg: float = 10.0     # roll from eye line
    head_tilt_warn_deg: float = 6.0

    fhd_depth_bad: float = 0.035      # nose forward vs shoulders in z (normalized units)
    fhd_depth_warn: float = 0.020

    # Shoulder openness via spacing (scale-invariant): shoulder_width / eye_distance
    shoulder_open_bad_ratio: float = 5.00
    shoulder_open_warn_ratio: float = 5.50

@dataclass
class Hysteresis:
    # stricter entering BAD, looser exiting to OK
    relax_deg: float = 3.0
    relax_ratio: float = 0.05

class PostureAnalyzer:
    def __init__(self, ema_alpha=0.35, thresholds: Thresholds = None, hysteresis: Hysteresis = None):
        self.a = ema_alpha
        self.thr = thresholds or Thresholds()
        self.hys = hysteresis or Hysteresis()

        # smoothed metrics
        self._neck_angle_v_deg_s = None
        self._fhd_ratio_s = None
        self._shoulder_slope_deg_s = None
        self._head_tilt_deg_s = None
        self._fhd_depth_s = None
        self._shoulder_open_ratio_s = None

        # sticky state for hysteresis
        self._bad_neck = False
        self._bad_fhd  = False
        self._bad_sho  = False
        self._bad_tilt = False

        self.last_analysis_ts = None

    # ---------- metric calculators ----------
    def _compute_neck_vertical_angle_deg(self, neck_base_px, nose_px):
        v_neck = _vec(neck_base_px, nose_px)
        v_up = (0.0, -1.0)
        return _angle_deg(v_neck, v_up)

    def _compute_forward_head_ratio(self, neck_base_px, nose_px, shoulder_width):
        if shoulder_width <= 1e-6: return 0.0
        return abs(nose_px[0] - neck_base_px[0]) / shoulder_width

    def _compute_shoulder_slope_deg(self, l_sho, r_sho):
        vec_sr = _vec(l_sho, r_sho)
        signed = math.degrees(math.atan2(vec_sr[1], vec_sr[0]))
        return abs(signed)

    def _compute_head_tilt_deg(self, l_eye, r_eye):
        v = _vec(l_eye, r_eye)
        deg = math.degrees(math.atan2(v[1], v[0]))
        return abs(deg)
    
    def _compute_forward_head_depth(self, nose, l_sho, r_sho):
        """Positive when nose is closer to camera than the shoulders (i.e., head forward)."""
        neck_z = (l_sho[2] + r_sho[2]) * 0.5
        return max(0.0, neck_z - nose[2])

    def _compute_shoulder_open_ratio(self, l_sho, r_sho, l_eye, r_eye):
        """Shoulder spacing normalized by eye distance (scale-invariant)."""
        shoulder_w = _len(_vec(l_sho, r_sho))
        eye_w = _len(_vec(l_eye, r_eye)) + 1e-6
        return shoulder_w / eye_w

    def _apply_hysteresis(self, val, bad_thr, warn_thr, was_bad, is_ratio=False):
        # When recovering, add slack (lower strictness)
        relax = self.hys.relax_ratio if is_ratio else self.hys.relax_deg
        bad_in  = bad_thr
        bad_out = (bad_thr - relax)
        warn_in = warn_thr
        warn_out = (warn_thr - relax)

        if was_bad:
            if val >= bad_out:
                return True, "BAD"
        # not bad: decide state by thresholds
        if val >= bad_in:
            return True, "BAD"
        elif val >= warn_in:
            return False, "WARN"
        else:
            return False, "OK"

    def analyze(self, pose_landmarks, image_shape):
        """
        Inputs:
          - pose_landmarks: MediaPipe PoseLandmarks
          - image_shape: (H, W, C)
        Returns:
          dict with metrics, statuses, and smoothed values.
        """
        H, W = image_shape[:2]

        # Required points present?
        try:
            lmk = pose_landmarks.landmark
            nose = _pt(lmk[POSE_NOSE], W, H)
            leye = _pt(lmk[POSE_L_EYE], W, H)
            reye = _pt(lmk[POSE_R_EYE], W, H)
            lsho = _pt(lmk[POSE_L_SHO], W, H)
            rsho = _pt(lmk[POSE_R_SHO], W, H)
        except Exception:
            return {"ok": False, "reason": "missing_landmarks"}

        # Derived anchors
        neck_base = ((lsho[0] + rsho[0]) * 0.5, (lsho[1] + rsho[1]) * 0.5)
        shoulder_width = _len(_vec(lsho, rsho))

        # Raw metrics
        neck_angle_v_deg = self._compute_neck_vertical_angle_deg(neck_base, nose)
        fhd_ratio        = self._compute_forward_head_ratio(neck_base, nose, shoulder_width)
        shoulder_slope   = self._compute_shoulder_slope_deg(lsho, rsho)
        head_tilt        = self._compute_head_tilt_deg(leye, reye)
        fhd_depth        = self._compute_forward_head_depth(nose, lsho, rsho)
        shoulder_open_r  = self._compute_shoulder_open_ratio(lsho, rsho, leye, reye)

        # Smooth
        self._neck_angle_v_deg_s   = _ema(self._neck_angle_v_deg_s,   neck_angle_v_deg, self.a)
        self._fhd_ratio_s          = _ema(self._fhd_ratio_s,          fhd_ratio,        self.a)
        self._shoulder_slope_deg_s = _ema(self._shoulder_slope_deg_s, shoulder_slope,   self.a)
        self._head_tilt_deg_s      = _ema(self._head_tilt_deg_s,      head_tilt,        self.a)
        self._fhd_depth_s          = _ema(self._fhd_depth_s,          fhd_depth,        self.a)
        self._shoulder_open_ratio_s= _ema(self._shoulder_open_ratio_s,shoulder_open_r,  self.a)

        # Hysteresis classification
        self._bad_neck, neck_state = self._apply_hysteresis(
            self._neck_angle_v_deg_s, self.thr.neck_flex_bad_deg, self.thr.neck_flex_warn_deg, self._bad_neck, is_ratio=False
        )
        
        # Combine 2D horizontal + depth forward into one state
        ratio_bad, ratio_state = self._apply_hysteresis(
            self._fhd_ratio_s, self.thr.fhd_bad_ratio, self.thr.fhd_warn_ratio, self._bad_fhd, is_ratio=True
        )
        depth_bad, depth_state = self._apply_hysteresis(
            self._fhd_depth_s, self.thr.fhd_depth_bad, self.thr.fhd_depth_warn, self._bad_fhd, is_ratio=True
        )

        if ratio_bad or depth_bad:
            self._bad_fhd = True
            fhd_state = "BAD"
        elif (ratio_state == "WARN") or (depth_state == "WARN"):
            self._bad_fhd = False
            fhd_state = "WARN"
        else:
            self._bad_fhd = False
            fhd_state = "OK"

        # Shoulder slope bands
        ang = float(self._shoulder_slope_deg_s) if self._shoulder_slope_deg_s is not None else 0.0
        if ang < 170.0:
            sho_state = "BAD"
            self._bad_sho = True
        elif ang < 175.5:
            sho_state = "WARN"
            self._bad_sho = False
        else:
            sho_state = "OK"
            self._bad_sho = False

        self._bad_tilt, tilt_state = self._apply_hysteresis(
            self._head_tilt_deg_s, self.thr.head_tilt_bad_deg, self.thr.head_tilt_warn_deg, self._bad_tilt, is_ratio=False
        )

        # Shoulder openness bands
        if self._shoulder_open_ratio_s is None:
            sho_open_state = "BAD"
        else:
            r = float(self._shoulder_open_ratio_s)
            if r < self.thr.shoulder_open_bad_ratio:
                sho_open_state = "BAD"
            elif r < self.thr.shoulder_open_warn_ratio:
                sho_open_state = "WARN"
            else:
                sho_open_state = "OK"

        overall_bad = (
            self._bad_neck or self._bad_fhd or self._bad_sho or self._bad_tilt
            or (sho_open_state == "BAD")
        )

        overall_state = "BAD" if overall_bad else (
                "WARN" if ("BAD" in [neck_state, fhd_state, sho_state, tilt_state, sho_open_state] or
                "WARN" in [neck_state, fhd_state, sho_state, tilt_state, sho_open_state]) else "OK"
        )

        self.last_analysis_ts = time.time()
        return {
            "ok": True,
            "state": overall_state,
            "metrics": {
                "neck_angle_deg": float(self._neck_angle_v_deg_s),
                "forward_head_ratio": float(self._fhd_ratio_s),
                "shoulder_slope_deg": float(self._shoulder_slope_deg_s),
                "head_tilt_deg": float(self._head_tilt_deg_s),
                "shoulder_width_px": float(shoulder_width),
                "fwd_head_depth": float(self._fhd_depth_s),
                "shoulder_open_ratio": float(self._shoulder_open_ratio_s),
            },
            "states": {
                "neck_flexion": neck_state,
                "forward_head": fhd_state,
                "shoulder_level": sho_state,
                "head_tilt": tilt_state,
                "shoulder_open": sho_open_state,
            },
            "points": {
                "neck_base_px": (float(neck_base[0]), float(neck_base[1])),
                "nose_px": (float(nose[0]), float(nose[1])),
                "l_shoulder_px": (float(lsho[0]), float(lsho[1])),
                "r_shoulder_px": (float(rsho[0]), float(rsho[1])),
                "l_eye_px": (float(leye[0]), float(leye[1])),
                "r_eye_px": (float(reye[0]), float(reye[1])),
            }
        }

    def draw_overlay(self, image, analysis):
        """Draw posture analysis overlay on image"""
        if not analysis.get("ok"):
            return

        h, w = image.shape[:2]
        m = analysis.get("metrics", {})
        s = analysis.get("states", {})
        overall = analysis.get("state", "OK")

        # Layout config
        margin = 16
        pad_x, pad_y = 16, 14
        body_scale = 0.85
        header_scale = 1.15
        body_th = 2
        header_th = 2
        line_gap = int(30 * body_scale)

        # Colors
        C_OK   = (80, 220, 100)
        C_WARN = (0, 215, 255)
        C_BAD  = (50, 50, 255)
        C_BG   = (0, 0, 0)

        def col(state):
            return C_OK if state == "OK" else C_WARN if state == "WARN" else C_BAD

        # Build lines
        lines = []
        lines.append((
            "Neck flex", f"{m.get('neck_angle_deg', 0.0):.1f}\u00B0", s.get("neck_flexion", "OK")
        ))
        lines.append((
            "Forward head (x)", f"{m.get('forward_head_ratio', 0.0):.2f}x", s.get("forward_head", "OK")
        ))
        if "fwd_head_depth" in m:
            lines.append((
                "Head fwd (z)", f"{m.get('fwd_head_depth', 0.0):.3f}", s.get("forward_head", "OK")
            ))
        lines.append((
            "Shoulder slope", f"{m.get('shoulder_slope_deg', 0.0):.1f}\u00B0", s.get("shoulder_level", "OK")
        ))
        if "shoulder_open_ratio" in m and "shoulder_open" in s:
            lines.append((
                "Shoulder open", f"{m.get('shoulder_open_ratio', 0.0):.2f}", s.get("shoulder_open", "OK")
            ))
        lines.append((
            "Head tilt", f"{m.get('head_tilt_deg', 0.0):.1f}\u00B0", s.get("head_tilt", "OK")
        ))

        # Measure text to size the card
        header_txt = f"POSTURE: {overall}"
        (hw, hh), _ = cv2.getTextSize(header_txt, cv2.FONT_HERSHEY_SIMPLEX, header_scale, header_th)

        max_line_w = 0
        for label, val, state in lines:
            txt = f"{label}: {val}  [{state}]"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, body_scale, body_th)
            max_line_w = max(max_line_w, tw)

        card_w = max(hw, max_line_w) + 2 * pad_x
        card_h = (pad_y + hh + 10) + (len(lines) * line_gap) + pad_y

        # Anchor at top-right
        x0 = w - card_w - margin
        y0 = margin
        x1 = x0 + card_w
        y1 = y0 + card_h

        # Draw card
        cv2.rectangle(image, (x0, y0), (x1, y1), C_BG, -1)
        cv2.rectangle(image, (x0, y0), (x1, y1), col(overall), 3)

        # Draw header
        hx = x0 + pad_x
        hy = y0 + pad_y + hh
        cv2.putText(image, header_txt, (hx, hy), cv2.FONT_HERSHEY_SIMPLEX, header_scale, col(overall), header_th, cv2.LINE_AA)

        # Draw lines
        y = hy + 12
        for label, val, state in lines:
            txt = f"{label}: {val}  [{state}]"
            y += line_gap
            cv2.putText(image, txt, (hx, y), cv2.FONT_HERSHEY_SIMPLEX, body_scale, col(state), body_th, cv2.LINE_AA)

# ======================= SYSTEM ACTIONS =======================

class _GammaRampDimmer:
    """Dims entire screen using SetDeviceGammaRamp (works on most Windows setups)."""
    def __init__(self):
        self._user32 = ctypes.WinDLL('user32', use_last_error=True)
        self._gdi32 = ctypes.WinDLL('gdi32', use_last_error=True)
        self._hdc = self._user32.GetDC(0)
        self._orig_ramp = (ctypes.c_ushort * 256 * 3)()
        self._got_orig = False
        self._got_orig = self._get_device_gamma_ramp(self._orig_ramp)
        self._dimmed = False

    def _get_device_gamma_ramp(self, ramp):
        fn = self._gdi32.GetDeviceGammaRamp
        fn.argtypes = [wintypes.HDC, ctypes.c_void_p]
        fn.restype = wintypes.BOOL
        return bool(fn(self._hdc, ctypes.byref(ramp)))

    def _set_device_gamma_ramp(self, ramp):
        fn = self._gdi32.SetDeviceGammaRamp
        fn.argtypes = [wintypes.HDC, ctypes.c_void_p]
        fn.restype = wintypes.BOOL
        return bool(fn(self._hdc, ctypes.byref(ramp)))

    def set_brightness(self, percent: float):
        """percent in [10..100]; lower = darker."""
        p = max(10.0, min(100.0, float(percent))) / 100.0
        ramp = (ctypes.c_ushort * 256 * 3)()
        for i in range(256):
            val = int(min(65535, max(0, int((i / 255.0) * 65535 * p))))
            ramp[0][i] = ramp[1][i] = ramp[2][i] = val
        ok = self._set_device_gamma_ramp(ramp)
        self._dimmed = ok
        return ok

    def restore(self):
        if self._got_orig:
            try:
                self._set_device_gamma_ramp(self._orig_ramp)
            except Exception:
                pass
        self._dimmed = False

def _notify(title: str, msg: str, duration=5):
    """Windows toast if available, else console print."""
    if _toaster:
        try:
            _toaster.show_toast(title, msg, duration=duration, threaded=True)
            return
        except Exception:
            pass
    print(f"[NOTIFY] {title}: {msg}")

class SystemActions:
    """Tracks timers and triggers actions."""
    def __init__(self, posture_bad_threshold_sec=180, phone_threshold_sec=180,
                 cooldown_sec=300, dim_percent=60):
        self.posture_bad_threshold = posture_bad_threshold_sec
        self.phone_threshold = phone_threshold_sec
        self.cooldown_sec = cooldown_sec
        self.dim_percent = dim_percent

        self._posture_bad_start = None
        self._phone_seen_start = None

        self._last_posture_alert = 0.0
        self._last_dim_action = 0.0

        self._dimmer = _GammaRampDimmer()
        self._dim_active = False

    @staticmethod
    def _count_bad_areas(analysis: dict) -> int:
        states = analysis.get("states", {})
        keys = ["neck_flexion", "forward_head", "head_tilt", "shoulder_level"]
        if "shoulder_open" in states:
            keys.append("shoulder_open")
        return sum(1 for k in keys if states.get(k) == "BAD")

    def update(self, analysis: dict, phones_detected_now: int):
        now = time.monotonic()

        # Posture timer (≥4 BAD areas)
        bad_count = self._count_bad_areas(analysis)
        if bad_count >= 4:
            if self._posture_bad_start is None:
                self._posture_bad_start = now
        else:
            self._posture_bad_start = None

        # Fire posture alert if threshold met and cooldown passed
        if self._posture_bad_start is not None:
            elapsed = now - self._posture_bad_start
            if elapsed >= self.posture_bad_threshold and (now - self._last_posture_alert) >= self.cooldown_sec:
                _notify("StraightUp", "Posture has been BAD in multiple areas for 3 minutes. Take a break and reset!")
                self._last_posture_alert = now

        # Phone timer (any phone visible)
        if (phones_detected_now or 0) > 0:
            if self._phone_seen_start is None:
                self._phone_seen_start = now
        else:
            self._phone_seen_start = None
            # If we were dimmed and phone gone, restore immediately
            if self._dim_active:
                self._dimmer.restore()
                self._dim_active = False

        # Dim screen if threshold met (and not already dim)
        if self._phone_seen_start is not None and not self._dim_active:
            elapsed = now - self._phone_seen_start
            if elapsed >= self.phone_threshold and (now - self._last_dim_action) >= self.cooldown_sec:
                ok = self._dimmer.set_brightness(self.dim_percent)
                self._dim_active = ok
                self._last_dim_action = now
                if ok:
                    _notify("StraightUp", "Phone on screen for 3 minutes. Dimming display.")
                else:
                    _notify("StraightUp", "Tried to dim display, but it wasn't supported.")

    def cleanup(self):
        # Always restore gamma if we changed it
        if self._dim_active:
            self._dimmer.restore()
            self._dim_active = False

# ======================= INTEGRATED DETECTOR =======================

class IntegratedPoseDetector:
    def __init__(self, enable_noise_detection=True):
        # Initialize MediaPipe solutions
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize posture analyzer and system actions
        self.posture = PostureAnalyzer(ema_alpha=0.35)
        self.system = SystemActions(
            posture_bad_threshold_sec=180,
            phone_threshold_sec=5,
            cooldown_sec=2,
            dim_percent=60,
        )
        
        # Phone usage tracking
        self.phone_usage_tracker = {
            'current_session': None,
            'usage_sessions': deque(maxlen=50),
            'total_usage_today': 0.0,
            'last_detection_time': 0.0,
            'session_start_time': None,
            'daily_stats': {'brief': 0, 'moderate': 0, 'extended': 0, 'excessive': 0}
        }
        
        # Focus and distraction analysis
        self.focus_score = 1.0
        self.distraction_factors = {
            'noise': 0.0,
            'phone_usage': 0.0,
            'posture': 0.0,
            'movement': 0.0
        }
        self.alerts = deque(maxlen=10)
        
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
        
        # Initialize enhanced phone detection system
        print("Loading enhanced phone detection models...")
        self.yolo_model = YOLO('yolo11n.pt')  # Base COCO model
        print("✅ YOLO COCO model loaded (class 67: cell phone)")
        
        # Phone detection configuration
        self.phone_confidence_threshold = 0.25
        self.phone_classes = [67]  # COCO cell phone class
        self.phone_detection_history = []
        self.max_history = 10
        
        # Enhanced phone tracking with smooth interpolation
        self._phone_tracks = []
        self._next_track_id = 1
        self.phone_smooth_alpha = 0.6
        self.phone_max_misses = 6
        
        # Phone detection smoothing to prevent flickering
        self.phone_smoothing_window = 5  # frames to consider
        self.phone_detection_threshold = 0.6  # 60% of frames must have phone
        self.min_session_duration = 1.0  # minimum 1 second before starting session
        self.session_end_delay = 2.0  # wait 2 seconds after last detection before ending
        self.last_stable_phone_time = 0.0
        self.session_candidate_start = None
        
        # EMA smoothing for neck center line
        self._neck_base_smooth = None
        self._head_pt_smooth = None
        
        print("📱 Enhanced phone detection with smooth tracking ready!")
        
        # Initialize MediaPipe models
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
    def _ema_smooth(self, prev, cur, a=0.3):
        """Exponential Moving Average for smooth positioning"""
        if prev is None:
            return cur
        return (int(a*cur[0] + (1-a)*prev[0]), int(a*cur[1] + (1-a)*prev[1]))
    
    # -------------------- Noise Detection Methods --------------------
    def toggle_noise_detection(self):
        """Toggle noise detection on/off"""
        if not self.noise_detector:
            print("⚠️  No noise detector available")
            return False
            
        if self.noise_enabled:
            return self.stop_noise_detection()
        else:
            return self.start_noise_detection()
    
    def start_noise_detection(self):
        """Start noise detection"""
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
    
    # -------------------- Phone Usage Tracking --------------------
    def update_phone_usage(self, phones_detected):
        """Update phone usage tracking with smart session management"""
        current_time = time.time()
        
        # Update detection history for smoothing
        self.phone_detection_history.append(phones_detected > 0)
        if len(self.phone_detection_history) > self.phone_smoothing_window:
            self.phone_detection_history.pop(0)
        
        # Calculate stability: what % of recent detections show phone?
        if len(self.phone_detection_history) >= self.phone_smoothing_window:
            phone_ratio = sum(self.phone_detection_history) / len(self.phone_detection_history)
            stable_phone_present = phone_ratio >= self.phone_detection_threshold
        else:
            stable_phone_present = phones_detected > 0
        
        # State machine for session management
        if stable_phone_present:
            self.last_stable_phone_time = current_time
            
            # Start new session if none exists
            if not self.phone_usage_tracker['current_session']:
                if not self.session_candidate_start:
                    self.session_candidate_start = current_time
                elif current_time - self.session_candidate_start >= self.min_session_duration:
                    # Start confirmed session
                    self.phone_usage_tracker['current_session'] = {
                        'start_time': self.session_candidate_start,
                        'end_time': None,
                        'duration': 0.0
                    }
                    self.phone_usage_tracker['session_start_time'] = self.session_candidate_start
                    print(f"📱 Phone session started at {time.strftime('%H:%M:%S')}")
        else:
            # No stable phone detection
            if self.session_candidate_start and not self.phone_usage_tracker['current_session']:
                # Cancel candidate session
                self.session_candidate_start = None
            
            # End existing session after delay
            if (self.phone_usage_tracker['current_session'] and 
                current_time - self.last_stable_phone_time >= self.session_end_delay):
                
                session = self.phone_usage_tracker['current_session']
                session['end_time'] = current_time
                session['duration'] = current_time - session['start_time']
                
                # Categorize session and add to history
                category = self._categorize_phone_session(session['duration'])
                self.phone_usage_tracker['daily_stats'][category] += 1
                self.phone_usage_tracker['usage_sessions'].append(session)
                self.phone_usage_tracker['total_usage_today'] += session['duration']
                
                # Show motivational message
                message = self.get_motivational_phone_message(category, session['duration'])
                print(f"📱 Session ended: {session['duration']:.1f}s ({category}) - {message}")
                
                # Update distraction factor
                self._update_phone_distraction_factor(category, session['duration'])
                
                # Clear current session
                self.phone_usage_tracker['current_session'] = None
                self.phone_usage_tracker['session_start_time'] = None
        
        # Update current session duration
        if self.phone_usage_tracker['current_session']:
            self.phone_usage_tracker['current_session']['duration'] = (
                current_time - self.phone_usage_tracker['current_session']['start_time']
            )
    
    def _categorize_phone_session(self, duration):
        """Categorize phone session by duration"""
        if duration < 10:
            return 'brief'
        elif duration < 30:
            return 'moderate'
        elif duration < 120:
            return 'extended'
        else:
            return 'excessive'
    
    def _update_phone_distraction_factor(self, category, duration):
        """Update phone usage distraction factor"""
        distraction_levels = {
            'brief': 0.1,
            'moderate': 0.3,
            'extended': 0.6,
            'excessive': 0.9
        }
        self.distraction_factors['phone_usage'] = distraction_levels.get(category, 0.0)
    
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
        return random.choice(messages.get(session_type, ['Stay focused! 🎯']))
    
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
        for i in range(len(points) - 1):
            cv2.line(image, points[i], points[i + 1], glow_color, thickness + 6)
        
        for i in range(len(points) - 1):
            cv2.line(image, points[i], points[i + 1], glow_color, thickness + 3)
        
        for i in range(len(points) - 1):
            cv2.line(image, points[i], points[i + 1], color, thickness)
    
    def draw_neon_circle(self, image, center, radius, color, glow_color):
        """Draw a neon glowing circle"""
        cv2.circle(image, center, radius + 6, glow_color, -1)
        cv2.circle(image, center, radius + 3, glow_color, -1)
        cv2.circle(image, center, radius, color, -1)
        cv2.circle(image, center, max(1, radius // 2), (255, 255, 255), -1)
    
    def draw_awesome_skeleton(self, image, landmarks):
        """Enhanced glowing skeleton with neck center line mapping"""
        h, w = image.shape[:2]
        
        # Get all landmark positions
        points = []
        for landmark in landmarks.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            points.append((x, y))
        
        # Define skeleton connections with different styles
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
            neck_base = self._ema_smooth(self._neck_base_smooth, neck_base_raw, a=0.7) if self._neck_base_smooth else neck_base_raw
            head_pt   = self._ema_smooth(self._head_pt_smooth,   head_pt_raw,   a=0.7) if self._head_pt_smooth   else head_pt_raw
            self._neck_base_smooth = neck_base
            self._head_pt_smooth   = head_pt

            # Draw the crisp, smooth center neck line
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
            [0, 1, 2, 3, 4],      # Thumb
            [0, 5, 6, 7, 8],      # Index finger  
            [0, 9, 10, 11, 12],   # Middle finger
            [0, 13, 14, 15, 16],  # Ring finger
            [0, 17, 18, 19, 20]   # Pinky
        ]
        
        # Draw each finger with glow
        for finger in connections:
            finger_points = [hand_points[i] for i in finger if i < len(hand_points)]
            if len(finger_points) > 1:
                self.draw_glow_effect(image, finger_points, self.HAND_COLOR, self.GLOW_HAND, 2)
        
        # Draw glowing fingertip points
        fingertips = [4, 8, 12, 16, 20]
        for tip_idx in fingertips:
            if tip_idx < len(hand_points):
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
                cv2.circle(image, (x, y), 1, self.EYE_COLOR, -1)
        
        if len(eye_points) > 3:
            eye_poly = np.array(eye_points, dtype=np.int32)
            cv2.polylines(image, [eye_poly], True, (150, 200, 255), 2)
            cv2.polylines(image, [eye_poly], True, self.EYE_COLOR, 1)
        
        # Draw iris with gentle animation
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
        """Enhanced phone detection with smooth tracking"""
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
        focus_color = (100, 100, 255)  # Default blue
        if self.focus_score > 0.7:
            focus_color = (0, 255, 0)  # Green for good focus
        elif self.focus_score < 0.4:
            focus_color = (0, 0, 255)  # Red for poor focus
        
        cv2.putText(image, f"Focus: {self.focus_score:.2f}", (20, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, focus_color, 2)
        
        # Noise detection status
        noise_status = "OFF"
        noise_color = (100, 100, 100)
        if self.noise_enabled and self.noise_detector:
            noise_status = "ON"
            noise_color = (0, 255, 0)
        
        cv2.putText(image, f"Noise: {noise_status}", (280, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, noise_color, 2)
        
        # Phone detection smoothing status
        if len(self.phone_detection_history) >= self.phone_smoothing_window:
            recent_detections = self.phone_detection_history[-self.phone_smoothing_window:]
            detection_ratio = sum(1 for x in recent_detections if x > 0) / len(recent_detections)
            smooth_color = (0, 255, 0) if detection_ratio >= self.phone_detection_threshold else (100, 100, 100)
            cv2.putText(image, f"Phone Smooth: {detection_ratio:.1f}", (280, 95), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, smooth_color, 2)
        
        # Session status
        session_status = "None"
        if self.phone_usage_tracker['current_session'] is not None:
            session_status = "Active"
        elif self.session_candidate_start is not None:
            session_status = "Candidate"
        
        cv2.putText(image, f"Session: {session_status}", (280, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Current session duration if active
        if self.phone_usage_tracker['current_session']:
            duration = self.phone_usage_tracker['current_session']['duration']
            cv2.putText(image, f"Duration: {duration:.1f}s", (280, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Daily phone usage stats
        stats = self.phone_usage_tracker['daily_stats']
        total_sessions = sum(stats.values())
        if total_sessions > 0:
            cv2.putText(image, f"Today: {total_sessions} sessions", (280, 160), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(image, f"Brief:{stats['brief']} Mod:{stats['moderate']}", (280, 175), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(image, f"Ext:{stats['extended']} Exc:{stats['excessive']}", (280, 190), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
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
            
            # Color based on session duration
            duration = session['duration']
            if duration < 10:
                color = (0, 255, 0)  # Green - brief
            elif duration < 30:
                color = (0, 255, 255)  # Yellow - moderate
            elif duration < 120:
                color = (0, 165, 255)  # Orange - extended
            else:
                color = (0, 0, 255)  # Red - excessive
            
            cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width - 1, y + height), color, -1)
        
        # Labels
        cv2.putText(image, "Phone Usage History", (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(image, f"Max: {max_duration:.0f}s", (x + width - 60, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
    
    def process_frame(self, image):
        """Process one frame: MediaPipe detection, smooth phone tracking, posture analysis, system actions"""
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

        # Posture analysis and system actions
        analysis = {"ok": False}
        if pose_detected:
            analysis = self.posture.analyze(pose_results.pose_landmarks, image.shape)
            self.posture.draw_overlay(image, analysis)

        # System actions (notifications / dimming)
        self.system.update(analysis, phones_detected)
        
        # Update phone usage tracking with smart session management
        self.update_phone_usage(phones_detected)
        
        # Analyze environmental factors for focus calculation
        self.analyze_environment()

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
        
        # Draw enhanced info panel with all statistics
        self.draw_enhanced_info_panel(image, faces_detected, hands_detected, pose_detected, phones_detected)
        
        # Draw noise indicators if enabled
        if self.noise_enabled and self.noise_detector:
            try:
                self.noise_detector.draw_noise_indicator(image, (580, 60))
                self.noise_detector.draw_noise_history(image, (580, 120), 200, 80)
            except Exception as e:
                # Silently handle noise drawing errors
                pass
        
        # Draw phone usage graph
        self.draw_phone_usage_graph(image, (580, 220), 200, 60)

        return image, faces_detected, hands_detected, pose_detected, phones_detected, analysis
    
    def run_webcam(self):
        """Run the integrated detection system on webcam"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("🎯 StraightUp Integrated Detection System Started!")
        print("=" * 60)
        print("✨ Integrated Features:")
        print("   👁️  Eyes: Natural contours + animated iris")
        print("   🤲 Hands: NEON skeleton with pulsing fingertips")
        print("   💪 Shoulders: HIGHLIGHTED with glow effects")
        print("   🏃 Body: ELECTRIC skeleton with animated joints")
        print("   📏 Neck: Smooth center line mapping (EMA smoothed)")
        print("   📱 Phones: Enhanced YOLO11n tracking with smoothing")
        print("   🧍 Posture: Real-time analysis with hysteresis")
        print("   � Noise: Environmental sound monitoring (press 'n')")
        print("   📊 Focus: Real-time distraction analysis")
        print("   �🔔 Notifications: Windows toast notifications")
        print("   🌙 Screen Dimming: Automatic gamma adjustment")
        print("   ✨ All with GLOW EFFECTS and smooth animations!")
        print("\n⌨️  Controls:")
        print("   'q' or ESC: Quit application")
        print("   's': Save current frame")
        print("   'i': Toggle info display")
        print("   'n': Toggle noise detection")
        print("   Space: Pause/Resume detection")
        print("=" * 60)
        
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
                    
                    # Process detections with integrated analysis
                    processed_frame, faces, hands, pose, phones, analysis = self.process_frame(frame)
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
                    cv2.rectangle(processed_frame, (10, 10), (450, 200), (0, 0, 0), -1)
                    cv2.rectangle(processed_frame, (10, 10), (450, 200), (255, 255, 255), 2)
                    
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
                    
                    # Posture status
                    if analysis.get("ok"):
                        posture_state = analysis.get("state", "UNKNOWN")
                        posture_color = (80, 220, 100) if posture_state == "OK" else (0, 215, 255) if posture_state == "WARN" else (50, 50, 255)
                        cv2.putText(processed_frame, f"Posture: {posture_state}", (20, 160), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, posture_color, 2)
                    
                    # System status
                    dim_status = "DIM" if self.system._dim_active else "NORMAL"
                    cv2.putText(processed_frame, f"Screen: {dim_status}", (20, 185), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    # Legend
                    legend_y = h - 100
                    cv2.rectangle(processed_frame, (10, legend_y - 10), (500, h - 10), (0, 0, 0), -1)
                    cv2.rectangle(processed_frame, (10, legend_y - 10), (500, h - 10), (255, 255, 255), 1)
                    
                    cv2.putText(processed_frame, "Eyes: Cyan + Red Iris", (20, legend_y + 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    cv2.putText(processed_frame, "Hands: Green Glow", (20, legend_y + 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(processed_frame, "Shoulders: Yellow Highlight", (20, legend_y + 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    cv2.putText(processed_frame, "Body: Blue/Pink Skeleton", (20, legend_y + 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    cv2.putText(processed_frame, "Phones: Magenta (Enhanced)", (280, legend_y + 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                    cv2.putText(processed_frame, "Neck: Yellow center line", (280, legend_y + 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    cv2.putText(processed_frame, "Posture: Real-time analysis", (280, legend_y + 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    cv2.putText(processed_frame, "System: Smart notifications", (280, legend_y + 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Pause indicator
                if paused:
                    h, w = processed_frame.shape[:2]
                    cv2.rectangle(processed_frame, (w//2 - 100, h//2 - 30), (w//2 + 100, h//2 + 30), (0, 0, 0), -1)
                    cv2.putText(processed_frame, "PAUSED", (w//2 - 80, h//2 + 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                
                cv2.imshow('StraightUp - Integrated Detection System', processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord('s'):  # Save frame
                    timestamp = int(time.time())
                    filename = f'integrated_capture_{timestamp}.jpg'
                    cv2.imwrite(filename, processed_frame)
                    print(f"💾 Frame saved as {filename}")
                elif key == ord('i'):  # Toggle info
                    show_info = not show_info
                    print(f"ℹ️  Info display: {'ON' if show_info else 'OFF'}")
                elif key == ord(' '):  # Space to pause
                    paused = not paused
                    print(f"⏸️  {'Paused' if paused else 'Resumed'}")
                elif key == ord('n'):  # Toggle noise detection
                    success = self.toggle_noise_detection()
                    status = "enabled" if self.noise_enabled else "disabled"
                    print(f"🔊 Noise detection {status}")
        
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Clean up noise detection
            if self.noise_enabled:
                self.stop_noise_detection()
            
            self.system.cleanup()
            print("✅ Integrated detection completed")

def main():
    """Main function to run the integrated detection system"""
    print("🚀 Starting StraightUp Integrated Detection System...")
    detector = IntegratedPoseDetector()
    detector.run_webcam()

if __name__ == "__main__":
    main()