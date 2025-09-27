# StraightUp 🎯

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
- `uv` package manager (recommended)

### Installation

**Option 1: Ultra-Fast with uv (Recommended)**
```bash
# Install uv if you don't have it
pip install uv

# Run immediately - dependencies auto-install!
uv run python detector.py
```

**Option 2: Traditional Setup**
```bash
# Install dependencies
pip install mediapipe opencv-python numpy ultralytics

# Run the application
python detector.py
```

## 🎮 Controls

While running:
- **'q' or ESC**: Quit application
- **'s'**: Save current frame with detections
- **'i'**: Toggle info display on/off
- **Space**: Pause/Resume detection

## 📁 Project Structure

```
StraightUp/
├── detector.py       # Main detection application
├── README.md             # This documentation
├── requirements.txt      # Dependencies
├── yolo11n.pt           # YOLO model (auto-downloaded)
└── pyproject.toml       # Modern Python project config
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