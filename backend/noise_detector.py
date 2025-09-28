"""
StraightUp - Audio Noise Detection System
Real-time noise level monitoring and analysis

Features:
- Real-time audio level monitoring
- Noise threshold detection
- Frequency analysis for noise classification
- Visual noise level indicators
- Integration with pose detection system
"""

import numpy as np
import pyaudio
import threading
import time
from collections import deque
import cv2
import matplotlib.pyplot as plt
from scipy import signal
import warnings
warnings.filterwarnings("ignore")


class NoiseDetector:
    """Real-time audio noise detection and analysis"""
    
    def __init__(self, sample_rate=44100, chunk_size=1024, channels=1):
        """Initialize noise detector
        
        Args:
            sample_rate (int): Audio sampling rate (Hz)
            chunk_size (int): Size of audio chunks to process
            channels (int): Number of audio channels (1=mono, 2=stereo)
        """
        # Audio configuration
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format = pyaudio.paInt16
        
        # PyAudio setup
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # Noise detection parameters
        self.noise_threshold = 0.02  # Adjustable noise threshold (0.0 to 1.0)
        self.quiet_threshold = 0.005  # Very quiet threshold
        self.loud_threshold = 0.1     # Very loud threshold
        
        # Data storage for analysis
        self.audio_buffer = deque(maxlen=100)  # Store last 100 chunks (~2-3 seconds)
        self.noise_history = deque(maxlen=300)  # Store noise levels for trending
        self.frequency_data = deque(maxlen=50)  # Store frequency analysis
        
        # Detection states
        self.is_noisy = False
        self.noise_level = 0.0
        self.peak_frequency = 0.0
        self.noise_category = "QUIET"
        
        # Threading control
        self.is_running = False
        self.detection_thread = None
        
        # Visual colors for noise levels (BGR format)
        self.QUIET_COLOR = (0, 255, 0)      # Green
        self.MODERATE_COLOR = (0, 165, 255)  # Orange  
        self.NOISY_COLOR = (0, 0, 255)      # Red
        self.VERY_LOUD_COLOR = (255, 0, 255) # Magenta
        
        print("🔊 Audio Noise Detector initialized")
        print(f"   Sample Rate: {sample_rate} Hz")
        print(f"   Chunk Size: {chunk_size} samples")
        print(f"   Channels: {channels}")
    
    def start_detection(self):
        """Start the noise detection in a separate thread"""
        if self.is_running:
            print("⚠️  Noise detection already running!")
            return
            
        try:
            # Open audio stream
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=None
            )
            
            self.is_running = True
            self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
            self.detection_thread.start()
            
            print("🎤 Noise detection started successfully!")
            
        except Exception as e:
            print(f"❌ Failed to start noise detection: {e}")
            self.is_running = False
    
    def stop_detection(self):
        """Stop the noise detection"""
        self.is_running = False
        
        if self.detection_thread:
            self.detection_thread.join(timeout=2.0)
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            
        print("🔇 Noise detection stopped")
    
    def _detection_loop(self):
        """Main detection loop - runs in separate thread"""
        print("🔄 Starting noise detection loop...")
        
        while self.is_running:
            try:
                # Read audio data
                if self.stream.is_active():
                    data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                    audio_array = np.frombuffer(data, dtype=np.int16)
                    
                    # Normalize to 0-1 range
                    if len(audio_array) > 0:
                        normalized_audio = audio_array.astype(np.float32) / 32768.0
                        self._process_audio_chunk(normalized_audio)
                
                time.sleep(0.01)  # Small delay to prevent excessive CPU usage
                
            except Exception as e:
                print(f"⚠️  Audio processing error: {e}")
                time.sleep(0.1)
    
    def _process_audio_chunk(self, audio_data):
        """Process a chunk of audio data"""
        # Calculate RMS (Root Mean Square) - represents overall volume
        rms = np.sqrt(np.mean(audio_data**2))
        self.noise_level = float(rms)
        
        # Store in buffer for analysis
        self.audio_buffer.append(audio_data)
        self.noise_history.append(self.noise_level)
        
        # Frequency analysis
        if len(audio_data) >= self.chunk_size:
            freqs, psd = signal.welch(audio_data, self.sample_rate, nperseg=min(512, len(audio_data)))
            
            # Find peak frequency
            if len(psd) > 0:
                peak_idx = np.argmax(psd)
                self.peak_frequency = freqs[peak_idx]
                self.frequency_data.append((freqs, psd))
        
        # Classify noise level
        self._classify_noise_level()
    
    def _classify_noise_level(self):
        """Classify current noise level"""
        level = self.noise_level
        
        if level < self.quiet_threshold:
            self.noise_category = "QUIET"
            self.is_noisy = False
        elif level < self.noise_threshold:
            self.noise_category = "MODERATE"
            self.is_noisy = False
        elif level < self.loud_threshold:
            self.noise_category = "NOISY"
            self.is_noisy = True
        else:
            self.noise_category = "VERY LOUD"
            self.is_noisy = True
    
    def get_noise_info(self):
        """Get current noise detection information"""
        return {
            'noise_level': self.noise_level,
            'is_noisy': self.is_noisy,
            'category': self.noise_category,
            'peak_frequency': self.peak_frequency,
            'threshold': self.noise_threshold,
            'history_length': len(self.noise_history)
        }
    
    def get_average_noise_level(self, seconds=5):
        """Get average noise level over the last N seconds"""
        if not self.noise_history:
            return 0.0
        
        # Approximate samples based on processing rate (~100 Hz)
        samples = min(len(self.noise_history), int(seconds * 100))
        recent_levels = list(self.noise_history)[-samples:]
        
        return np.mean(recent_levels) if recent_levels else 0.0
    
    def is_consistently_noisy(self, seconds=3, threshold_ratio=0.7):
        """Check if environment has been consistently noisy"""
        if len(self.noise_history) < 10:
            return False
        
        samples = min(len(self.noise_history), int(seconds * 100))
        recent_levels = list(self.noise_history)[-samples:]
        
        noisy_count = sum(1 for level in recent_levels if level > self.noise_threshold)
        noisy_ratio = noisy_count / len(recent_levels)
        
        return noisy_ratio >= threshold_ratio
    
    def adjust_sensitivity(self, sensitivity_level):
        """Adjust noise detection sensitivity
        
        Args:
            sensitivity_level (str): 'low', 'medium', 'high', or 'very_high'
        """
        sensitivity_map = {
            'low': {'quiet': 0.01, 'noise': 0.05, 'loud': 0.15},
            'medium': {'quiet': 0.005, 'noise': 0.02, 'loud': 0.1},
            'high': {'quiet': 0.003, 'noise': 0.01, 'loud': 0.05},
            'very_high': {'quiet': 0.001, 'noise': 0.005, 'loud': 0.02}
        }
        
        if sensitivity_level in sensitivity_map:
            settings = sensitivity_map[sensitivity_level]
            self.quiet_threshold = settings['quiet']
            self.noise_threshold = settings['noise']
            self.loud_threshold = settings['loud']
            
            print(f"🎚️  Sensitivity set to {sensitivity_level.upper()}")
            print(f"   Quiet: < {self.quiet_threshold:.3f}")
            print(f"   Noise: < {self.noise_threshold:.3f}")
            print(f"   Loud:  < {self.loud_threshold:.3f}")
        else:
            print(f"❌ Unknown sensitivity level: {sensitivity_level}")
    
    def draw_noise_indicator(self, image, position=(50, 50)):
        """Draw noise level indicator on image
        
        Args:
            image: OpenCV image to draw on
            position: (x, y) position for the indicator
        """
        if not self.is_running:
            return
        
        x, y = position
        
        # Choose color based on noise level
        if self.noise_category == "QUIET":
            color = self.QUIET_COLOR
        elif self.noise_category == "MODERATE":
            color = self.MODERATE_COLOR
        elif self.noise_category == "NOISY":
            color = self.NOISY_COLOR
        else:
            color = self.VERY_LOUD_COLOR
        
        # Draw noise level bar
        bar_width = 200
        bar_height = 20
        fill_width = int(bar_width * min(self.noise_level / 0.1, 1.0))
        
        # Background bar
        cv2.rectangle(image, (x, y), (x + bar_width, y + bar_height), (50, 50, 50), -1)
        
        # Fill bar
        if fill_width > 0:
            cv2.rectangle(image, (x, y), (x + fill_width, y + bar_height), color, -1)
        
        # Border
        cv2.rectangle(image, (x, y), (x + bar_width, y + bar_height), (255, 255, 255), 2)
        
        # Labels
        cv2.putText(image, f"NOISE: {self.noise_category}", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(image, f"{self.noise_level:.3f}", (x + bar_width + 10, y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Threshold line
        threshold_x = x + int(bar_width * (self.noise_threshold / 0.1))
        cv2.line(image, (threshold_x, y), (threshold_x, y + bar_height), (255, 255, 255), 2)
        
        # Frequency info (if available)
        if self.peak_frequency > 0:
            freq_text = f"{self.peak_frequency:.0f} Hz"
            cv2.putText(image, freq_text, (x, y + bar_height + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def draw_noise_history(self, image, position=(50, 100), width=300, height=100):
        """Draw noise level history graph
        
        Args:
            image: OpenCV image to draw on
            position: (x, y) position for the graph
            width: Graph width in pixels
            height: Graph height in pixels
        """
        if not self.noise_history or len(self.noise_history) < 2:
            return
        
        x, y = position
        
        # Background
        cv2.rectangle(image, (x, y), (x + width, y + height), (30, 30, 30), -1)
        cv2.rectangle(image, (x, y), (x + width, y + height), (255, 255, 255), 1)
        
        # Get recent history
        history = list(self.noise_history)[-width:]  # Last 'width' samples
        if len(history) < 2:
            return
        
        # Normalize to graph height
        max_level = max(max(history), self.loud_threshold)
        
        # Draw history line
        points = []
        for i, level in enumerate(history):
            graph_x = x + i
            graph_y = y + height - int((level / max_level) * height)
            points.append((graph_x, graph_y))
        
        # Draw the line
        if len(points) > 1:
            for i in range(len(points) - 1):
                color = self.QUIET_COLOR
                if history[i] > self.loud_threshold:
                    color = self.VERY_LOUD_COLOR
                elif history[i] > self.noise_threshold:
                    color = self.NOISY_COLOR
                elif history[i] > self.quiet_threshold:
                    color = self.MODERATE_COLOR
                
                cv2.line(image, points[i], points[i + 1], color, 2)
        
        # Draw threshold lines
        threshold_y = y + height - int((self.noise_threshold / max_level) * height)
        cv2.line(image, (x, threshold_y), (x + width, threshold_y), (255, 255, 0), 1)
        
        loud_threshold_y = y + height - int((self.loud_threshold / max_level) * height)
        cv2.line(image, (x, loud_threshold_y), (x + width, loud_threshold_y), (255, 0, 0), 1)
        
        # Labels
        cv2.putText(image, "Noise History", (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def get_noise_alerts(self):
        """Get noise-based alerts and recommendations"""
        alerts = []
        
        if self.is_consistently_noisy(seconds=5):
            alerts.append({
                'type': 'warning',
                'message': 'Environment has been consistently noisy for 5+ seconds',
                'recommendation': 'Consider moving to a quieter location'
            })
        
        if self.noise_level > self.loud_threshold:
            alerts.append({
                'type': 'critical',
                'message': f'Very loud noise detected ({self.noise_level:.3f})',
                'recommendation': 'Immediate attention required - check surroundings'
            })
        
        avg_5sec = self.get_average_noise_level(5)
        if avg_5sec > self.noise_threshold * 2:
            alerts.append({
                'type': 'info',
                'message': f'High average noise level: {avg_5sec:.3f}',
                'recommendation': 'Consider noise-canceling measures'
            })
        
        return alerts
    
    def __del__(self):
        """Cleanup when detector is destroyed"""
        if self.is_running:
            self.stop_detection()
        
        if hasattr(self, 'audio'):
            self.audio.terminate()


def demo_noise_detector():
    """Demo function to test noise detector standalone"""
    print("🎤 Starting Noise Detector Demo...")
    
    detector = NoiseDetector()
    detector.start_detection()
    
    # Create a simple visualization window
    try:
        print("📺 Opening visualization window...")
        print("Press 'q' to quit, 's' to change sensitivity")
        
        sensitivity_levels = ['low', 'medium', 'high', 'very_high']
        current_sensitivity = 1  # Start with 'medium'
        
        while True:
            # Create a black canvas
            canvas = np.zeros((400, 600, 3), dtype=np.uint8)
            
            # Draw noise indicator
            detector.draw_noise_indicator(canvas, (50, 50))
            
            # Draw noise history
            detector.draw_noise_history(canvas, (50, 150), 500, 100)
            
            # Show current settings
            sensitivity_text = f"Sensitivity: {sensitivity_levels[current_sensitivity].upper()}"
            cv2.putText(canvas, sensitivity_text, (50, 300),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show noise info
            info = detector.get_noise_info()
            info_text = f"Peak Freq: {info['peak_frequency']:.0f} Hz"
            cv2.putText(canvas, info_text, (50, 330),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Show alerts
            alerts = detector.get_noise_alerts()
            for i, alert in enumerate(alerts[:3]):  # Show max 3 alerts
                alert_color = (0, 255, 255) if alert['type'] == 'warning' else (0, 0, 255)
                cv2.putText(canvas, alert['message'][:50], (50, 360 + i * 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, alert_color, 1)
            
            cv2.imshow('Noise Detector Demo', canvas)
            
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                current_sensitivity = (current_sensitivity + 1) % len(sensitivity_levels)
                detector.adjust_sensitivity(sensitivity_levels[current_sensitivity])
    
    except KeyboardInterrupt:
        print("\n🛑 Demo stopped by user")
    
    finally:
        detector.stop_detection()
        cv2.destroyAllWindows()
        print("✅ Demo completed")


if __name__ == "__main__":
    demo_noise_detector()