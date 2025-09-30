# StraightUp 🎯

**Pose Detection System with Computer Vision & Noise Monitoring**

A real-time detection system that combines MediaPipe and YOLO11n for human pose analysis with noise monitoring.

## ✨ Features

### 🎥 Visual Detection
- **👁️ Eyes:** Eye tracking with iris detection
- **🤲 Hands:** Hand skeleton with fingertip detection (21 points)
- **💪 Shoulders:** Shoulder position highlighting
- **🏃 Body:** Full body pose detection (33 points)
- **🔗 Neck:** Neck center line mapping
- **📱 Phones:** Phone detection with tracking
- **✨ Visual effects and smooth animations**

### 🔊 Audio Noise Detection
- **🎤 Real-time Audio Monitoring:** Ambient noise level analysis
- **🎚️ Adjustable Sensitivity:** Low, medium, high, and very high sensitivity modes
- **📊 Visual Indicators:** Live noise level bars and history graphs
- **⚠️ Smart Alerts:** Notifications for noise events
- **📈 Frequency Analysis:** Peak frequency detection and classification
- **🎯 Focus Analysis:** Focus score based on environmental factors

### 📱 Smart Phone Alerts
- **⏱️ Usage Tracking:** Phone usage session monitoring
- **🎯 Smart Notifications:** Alerts for different usage patterns
- **💪 Motivational Messages:** Feedback for good habits
- **📊 Usage Analytics:** Statistics and usage pattern analysis
- **🎨 Visual Feedback:** Usage history graphs and productivity scores
- **☕ Break Suggestions:** Recommendations for healthy breaks

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Webcam
- Microphone (for noise detection)
- `uv` package manager (recommended)

### Installation

**Option 1: Ultra-Fast with uv (Recommended)**
```bash
# Install uv if you don't have it
pip install uv

# Test noise detection first
uv run python test_noise.py

# Test phone alert system
uv run python test_phone_alerts.py

# Quick noise demo
uv run python quick_noise_demo.py

# Run basic detection (no noise monitoring)
uv run python detector.py

# Run enhanced detection with noise & phone alerts
uv run python enhanced_detector.py
```

**Option 2: Traditional Setup**
```bash
# Install basic dependencies
pip install mediapipe opencv-python numpy ultralytics

# Install noise detection dependencies
pip install pyaudio scipy matplotlib

# Run the applications
python detector.py           # Basic detection
python enhanced_detector.py  # With noise monitoring
```

**Option 3: Windows PyAudio Fix (if needed)**
```bash
# If PyAudio installation fails on Windows
pip install pipwin
pipwin install pyaudio
```

## 🎮 Controls

### Basic Detection (detector.py)
- **'q' or ESC**: Quit application
- **'s'**: Save current frame with detections
- **'i'**: Toggle info display on/off
- **Space**: Pause/Resume detection

### Enhanced Detection (enhanced_detector.py)
- **'q' or ESC**: Quit application
- **'s'**: Save current frame with detections
- **'i'**: Toggle info display on/off
- **'n'**: Toggle noise detection on/off
- **'a'**: Toggle alert display on/off
- **'p'**: Reset phone usage statistics
- **'r'**: Show detailed phone usage report
- **Space**: Pause/Resume detection

## 📁 Project Structure

```
StraightUp/backend/
├── detector.py           # Main detection application (basic)
├── enhanced_detector.py  # Enhanced detection with noise & phone alerts
├── noise_detector.py     # Standalone noise detection module
├── test_noise.py         # Test script for noise detection setup
├── test_phone_alerts.py  # Test script for phone alert system
├── quick_noise_demo.py   # Simple noise detection demo
├── README.md             # This documentation
├── pyproject.toml        # Modern Python project config
├── uv.lock              # Dependency lock file
└── yolo11n.pt           # YOLO model (auto-downloaded)
```

## � Technical Details

**AI Models:**
- **MediaPipe:** Face mesh (468 landmarks), hands (21 points each), pose (33 points)
- **YOLO11n:** Phone detection with confidence scoring
- **EMA Smoothing:** Neck center line and phone tracking

**Performance:**
- **Resolution:** 1280x720 (HD)
- **FPS:** 60+ target with optimizations
- **Platform:** Cross-platform (Windows, Mac, Linux)
- **Hardware:** CPU-based inference (no GPU required)

**Visual Effects:**
- Multi-layer glow effects with neon colors
- Animated pulsing joints and fingertips
- Smooth EMA-based neck center line mapping
- Real-time FPS display and detection counts

## ⚡ Key Features

- **Real-time Processing:** Optimized for live webcam input
- **Multi-person Support:** Up to 2 faces and 2 hands simultaneously
- **Enhanced Phone Detection:** YOLO11n with smooth tracking
- **Neck Mapping:** Unique center line feature with EMA smoothing
- **Modern Tooling:** Uses latest Python packaging standards
- **No GPU Required:** Efficient CPU-based processing

## 🛠️ Troubleshooting

**Camera Issues:**
- Ensure no other applications are using the webcam
- Check camera permissions in system settings
- Try different camera index if multiple cameras available

**Installation Issues:**
- Install uv first: `pip install uv`
- Use uv for automatic dependency management: `uv run python detector.py`
- Check Python version: `python --version` (should be 3.11+)

## 📊 System Requirements

- **Python:** 3.11 or higher
- **RAM:** 4GB minimum, 8GB recommended
- **CPU:** Modern multi-core processor
- **Camera:** Any USB webcam or built-in camera
- **OS:** Windows 10+, macOS 10.14+, or Linux

## 🎯 Use Cases

- **Fitness Applications:** Real-time posture analysis
- **Accessibility Tools:** Hand gesture recognition
- **Security Systems:** Phone usage detection
- **Research Projects:** Human pose analysis
- **Educational Demos:** Computer vision learning

## 📝 Notes

This project uses Google's MediaPipe framework for human pose analysis combined with Ultralytics YOLO11n for object detection. The visual effects and tracking make it good for demonstrations and real-world applications.

**Built for modern computer vision applications** 🚀

## 📚 Resources

- [MediaPipe Documentation](https://mediapipe.dev/)
- [Ultralytics YOLO](https://ultralytics.com/)
- [uv Package Manager](https://github.com/astral-sh/uv)