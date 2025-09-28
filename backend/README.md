# StraightUp 🎯

**Enhanced Pose Detection System with Real-time Computer Vision & Noise Monitoring**

A compr## 🔊 Technical Details

**AI Models:**
- **MediaPipe:** Face mesh (468 landmarks), hands (21 points each), pose (33 points)
- **YOLO11n:** Phone detection with confidence scoring
- **EMA Smoothing:** Neck center line and phone tracking

**Audio Processing:**
- **Sample Rate:** 44.1kHz with 1024-sample chunks
- **Analysis:** Real-time RMS level calculation and frequency analysis
- **Smoothing:** Exponential moving average for consistent readings
- **Threading:** Non-blocking audio processing in separate thread

**Performance:**
- **Resolution:** 1280x720 (HD)
- **FPS:** 60+ target with optimizations
- **Audio Latency:** <50ms for real-time responsiveness
- **Platform:** Cross-platform (Windows, Mac, Linux)
- **Hardware:** CPU-based inference (no GPU required)

**Visual Effects:**
- Multi-layer glow effects with neon colors
- Animated pulsing joints and fingertips
- Smooth EMA-based neck center line mapping
- Real-time noise level visualization
- Focus score and distraction analysis displays detection system that combines MediaPipe, YOLO11n, and audio noise detection for advanced human pose analysis with environmental awareness and stunning visual effects.

## ✨ Features

### 🎥 Visual Detection
- **👁️ Eyes:** Natural contours with animated iris tracking
- **🤲 Hands:** NEON skeleton with pulsing fingertips (21-point detection)
- **💪 Shoulders:** Highlighted with glow effects and labels
- **🏃 Body:** Electric skeleton with animated joints (33-point pose)
- **🔗 Neck:** Smooth center line mapping with EMA smoothing
- **📱 Phones:** Enhanced YOLO11n detection with smooth tracking
- **✨ All with multi-layer glow effects and smooth animations**

### 🔊 Audio Noise Detection
- **🎤 Real-time Audio Monitoring:** Continuous ambient noise level analysis
- **🎚️ Adjustable Sensitivity:** Low, medium, high, and very high sensitivity modes
- **📊 Visual Indicators:** Live noise level bars and history graphs
- **⚠️ Smart Alerts:** Context-aware notifications for noise events
- **📈 Frequency Analysis:** Peak frequency detection and classification
- **🎯 Focus Analysis:** AI-powered focus score based on environmental factors

### 📱 Smart Phone Alerts
- **⏱️ Usage Tracking:** Real-time phone usage session monitoring
- **🎯 Smart Notifications:** Context-aware alerts for different usage patterns
- **💪 Motivational Messages:** Encouraging feedback for good habits
- **📊 Usage Analytics:** Detailed statistics and usage pattern analysis
- **🎨 Visual Feedback:** Usage history graphs and productivity scores
- **☕ Break Suggestions:** Intelligent recommendations for healthy breaks

**Enhanced Pose Detection System with Real-time Computer Vision**

A comprehensive real-time detection system that combines MediaPipe and YOLO11n for advanced human pose analysis with stunning visual effects.

## ✨ Features

- **👁️ Eyes:** Natural contours with animated iris tracking
- **🤲 Hands:** NEON skeleton with pulsing fingertips (21-point detection)
- **💪 Shoulders:** Highlighted with glow effects and labels
- **🏃 Body:** Electric skeleton with animated joints (33-point pose)
- **🔗 Neck:** Smooth center line mapping with EMA smoothing
- **� Phones:** Enhanced YOLO11n detection with smooth tracking
- **✨ All with multi-layer glow effects and smooth animations**

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

This project leverages Google's MediaPipe framework for comprehensive human pose analysis combined with Ultralytics YOLO11n for object detection. The enhanced visual effects and smooth tracking make it perfect for demonstrations and real-world applications.

**Built for modern computer vision applications** 🚀

## 📚 Resources

- [MediaPipe Documentation](https://mediapipe.dev/)
- [Ultralytics YOLO](https://ultralytics.com/)
- [uv Package Manager](https://github.com/astral-sh/uv)