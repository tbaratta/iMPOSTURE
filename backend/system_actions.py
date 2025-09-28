"""
StraightUp - System Actions Module
Handles notifications, screen dimming, and system interactions
"""

import cv2
import numpy as np
import time
import os
import sys
from typing import Dict, Any, Optional, List


class SystemActions:
    """Manages system notifications and actions"""
    
    def __init__(self):
        self.notifications_enabled = True
        self.screen_dimming_enabled = True
        self.last_notification_time = {}
        self.notification_cooldown = 30.0  # seconds
        
        # Focus scoring system
        self.focus_metrics = {
            'human_present': 0.0,
            'good_posture': 0.0,
            'no_phone': 0.0,
            'quiet_environment': 0.0,
            'overall_score': 0.0
        }
        
        self.focus_history = []
        self.focus_smoothing_alpha = 0.1
        
        # Screen dimming configuration
        self.screen_dim_level = 30  # Percentage brightness when dimmed
        self.dim_delay = 60.0  # seconds of poor focus before dimming
        self.poor_focus_start_time = None
        self.screen_dimmed = False
        
        print("⚙️ System actions module ready!")
    
    def send_windows_notification(self, title, message, icon_type="info"):
        """Send Windows toast notification"""
        if not self.notifications_enabled:
            return False
        
        # Check cooldown
        notification_key = f"{title}:{message}"
        current_time = time.time()
        if (notification_key in self.last_notification_time and 
            current_time - self.last_notification_time[notification_key] < self.notification_cooldown):
            return False
        
        try:
            # Try using win10toast (more reliable for Windows 10/11)
            try:
                from win10toast import ToastNotifier
                toaster = ToastNotifier()
                toaster.show_toast(
                    title,
                    message,
                    duration=5,
                    threaded=True
                )
                self.last_notification_time[notification_key] = current_time
                return True
            except ImportError:
                pass
            
            # Fallback to plyer
            try:
                from plyer import notification
                notification.notify(
                    title=title,
                    message=message,
                    timeout=5
                )
                self.last_notification_time[notification_key] = current_time
                return True
            except ImportError:
                pass
            
            # Final fallback - command line notification
            if sys.platform == "win32":
                import subprocess
                # Use PowerShell for Windows notifications
                ps_script = f'''
                Add-Type -AssemblyName System.Windows.Forms
                $notification = New-Object System.Windows.Forms.NotifyIcon
                $notification.Icon = [System.Drawing.SystemIcons]::Information
                $notification.BalloonTipIcon = "Info"
                $notification.BalloonTipText = "{message}"
                $notification.BalloonTipTitle = "{title}"
                $notification.Visible = $true
                $notification.ShowBalloonTip(5000)
                Start-Sleep -Seconds 6
                $notification.Dispose()
                '''
                subprocess.Popen(['powershell', '-Command', ps_script], 
                               shell=True, creationflags=subprocess.CREATE_NO_WINDOW)
                self.last_notification_time[notification_key] = current_time
                return True
            
        except Exception as e:
            print(f"⚠️ Notification error: {e}")
            return False
        
        return False
    
    def dim_screen(self, dim_level=None):
        """Dim the screen brightness (Windows)"""
        if not self.screen_dimming_enabled:
            return False
        
        if dim_level is None:
            dim_level = self.screen_dim_level
        
        try:
            if sys.platform == "win32":
                import subprocess
                # Use PowerShell to adjust brightness
                ps_script = f'''
                (Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1,{dim_level})
                '''
                subprocess.run(['powershell', '-Command', ps_script], 
                             shell=True, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
                self.screen_dimmed = True
                return True
        except Exception as e:
            print(f"⚠️ Screen dimming error: {e}")
        
        return False
    
    def restore_screen_brightness(self):
        """Restore normal screen brightness"""
        try:
            if sys.platform == "win32" and self.screen_dimmed:
                import subprocess
                # Restore to 100% brightness
                ps_script = '''
                (Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1,100)
                '''
                subprocess.run(['powershell', '-Command', ps_script], 
                             shell=True, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
                self.screen_dimmed = False
                return True
        except Exception as e:
            print(f"⚠️ Screen restore error: {e}")
        
        return False
    
    def update_focus_score(self, human_detected, posture_score, phones_detected, noise_level):
        """Update focus scoring metrics"""
        current_time = time.time()
        
        # Calculate individual metric scores (0.0 to 1.0)
        human_score = 1.0 if human_detected else 0.0
        posture_score_normalized = max(0.0, min(1.0, posture_score / 100.0)) if posture_score is not None else 0.5
        phone_score = 1.0 if phones_detected == 0 else max(0.0, 1.0 - (phones_detected * 0.5))
        noise_score = 1.0 if noise_level is None else max(0.0, 1.0 - max(0.0, (noise_level - 0.3) / 0.4))
        
        # Apply smoothing
        alpha = self.focus_smoothing_alpha
        self.focus_metrics['human_present'] = (
            self.focus_metrics['human_present'] * (1 - alpha) + human_score * alpha
        )
        self.focus_metrics['good_posture'] = (
            self.focus_metrics['good_posture'] * (1 - alpha) + posture_score_normalized * alpha
        )
        self.focus_metrics['no_phone'] = (
            self.focus_metrics['no_phone'] * (1 - alpha) + phone_score * alpha
        )
        self.focus_metrics['quiet_environment'] = (
            self.focus_metrics['quiet_environment'] * (1 - alpha) + noise_score * alpha
        )
        
        # Calculate overall score (weighted average)
        weights = {
            'human_present': 0.2,
            'good_posture': 0.3,
            'no_phone': 0.3,
            'quiet_environment': 0.2
        }
        
        overall_score = sum(
            self.focus_metrics[metric] * weight 
            for metric, weight in weights.items()
        )
        
        self.focus_metrics['overall_score'] = overall_score
        
        # Track focus history
        self.focus_history.append({
            'timestamp': current_time,
            'score': overall_score,
            'metrics': self.focus_metrics.copy()
        })
        
        # Limit history size
        if len(self.focus_history) > 1800:  # 1 minute at 30fps
            self.focus_history.pop(0)
        
        # Handle poor focus tracking for screen dimming
        if overall_score < 0.6:  # Poor focus threshold
            if self.poor_focus_start_time is None:
                self.poor_focus_start_time = current_time
            elif (current_time - self.poor_focus_start_time > self.dim_delay and 
                  not self.screen_dimmed):
                self.dim_screen()
                self.send_windows_notification(
                    "StraightUp Focus Alert",
                    "Poor focus detected - screen dimmed to help you refocus",
                    "warning"
                )
        else:
            # Good focus - reset timer and restore screen
            self.poor_focus_start_time = None
            if self.screen_dimmed:
                self.restore_screen_brightness()
        
        return overall_score
    
    def check_posture_alerts(self, posture_data):
        """Check for posture-related alerts"""
        if not posture_data:
            return
        
        current_time = time.time()
        score = posture_data.get('overall_score', 100)
        
        # Poor posture alert
        if score < 60:
            self.send_windows_notification(
                "StraightUp Posture Alert",
                f"Poor posture detected (Score: {score:.0f}/100). Time to adjust!",
                "warning"
            )
        
        # Check individual issues
        if posture_data.get('neck_strained'):
            self.send_windows_notification(
                "StraightUp Neck Alert",
                "Neck strain detected! Roll your shoulders and adjust your screen height.",
                "warning"
            )
        
        if posture_data.get('head_forward'):
            self.send_windows_notification(
                "StraightUp Head Position",
                "Forward head posture detected. Pull your head back and align with shoulders.",
                "info"
            )
    
    def check_phone_alerts(self, phone_usage_data):
        """Check for phone usage alerts"""
        if not phone_usage_data:
            return
        
        current_session = phone_usage_data.get('current_session')
        if current_session and current_session.get('duration', 0) > 60:  # 1 minute
            duration = current_session['duration']
            if duration > 300:  # 5 minutes
                self.send_windows_notification(
                    "StraightUp Phone Alert",
                    f"Extended phone use: {duration//60:.0f}m {duration%60:.0f}s. Time for a break?",
                    "warning"
                )
            elif duration > 120:  # 2 minutes
                self.send_windows_notification(
                    "StraightUp Phone Notice",
                    f"Phone session: {duration//60:.0f}m {duration%60:.0f}s. Consider wrapping up.",
                    "info"
                )
    
    def check_noise_alerts(self, noise_level):
        """Check for noise-related alerts"""
        if noise_level is None:
            return
        
        if noise_level > 0.8:  # Very noisy
            self.send_windows_notification(
                "StraightUp Environment",
                "Very noisy environment detected. Consider using headphones or finding a quieter space.",
                "info"
            )
        elif noise_level > 0.6:  # Moderately noisy
            self.send_windows_notification(
                "StraightUp Environment",
                "Moderate noise detected. This might affect your concentration.",
                "info"
            )
    
    def get_focus_category(self, score):
        """Get focus category based on score"""
        if score >= 0.8:
            return 'excellent'
        elif score >= 0.6:
            return 'good'
        elif score >= 0.4:
            return 'fair'
        else:
            return 'poor'
    
    def get_focus_color(self, score):
        """Get color for focus score visualization"""
        if score >= 0.8:
            return (0, 255, 0)    # Green
        elif score >= 0.6:
            return (0, 255, 255)  # Yellow
        elif score >= 0.4:
            return (0, 165, 255)  # Orange
        else:
            return (0, 0, 255)    # Red
    
    def draw_focus_metrics(self, image, position=(580, 400), width=200, height=120):
        """Draw focus metrics display"""
        x, y = position
        
        # Background
        cv2.rectangle(image, (x, y), (x + width, y + height), (30, 30, 30), -1)
        cv2.rectangle(image, (x, y), (x + width, y + height), (255, 255, 255), 1)
        
        # Title
        cv2.putText(image, "Focus Metrics", (x + 5, y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Overall score
        overall_score = self.focus_metrics['overall_score']
        score_color = self.get_focus_color(overall_score)
        cv2.putText(image, f"Overall: {overall_score:.2f}", (x + 5, y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, score_color, 1)
        
        # Individual metrics
        metrics_display = [
            ("Human", self.focus_metrics['human_present']),
            ("Posture", self.focus_metrics['good_posture']),
            ("No Phone", self.focus_metrics['no_phone']),
            ("Quiet", self.focus_metrics['quiet_environment'])
        ]
        
        for i, (name, value) in enumerate(metrics_display):
            y_pos = y + 55 + (i * 15)
            color = self.get_focus_color(value)
            
            # Metric name
            cv2.putText(image, f"{name}:", (x + 5, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Progress bar
            bar_width = 80
            bar_x = x + 65
            cv2.rectangle(image, (bar_x, y_pos - 8), (bar_x + bar_width, y_pos - 2), (60, 60, 60), -1)
            filled_width = int(bar_width * value)
            cv2.rectangle(image, (bar_x, y_pos - 8), (bar_x + filled_width, y_pos - 2), color, -1)
            
            # Value text
            cv2.putText(image, f"{value:.2f}", (bar_x + bar_width + 5, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Screen dim status
        if self.screen_dimmed:
            cv2.putText(image, "🔅 Screen Dimmed", (x + 5, y + height - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    def draw_focus_history(self, image, position=(580, 530), width=200, height=60):
        """Draw focus score history graph"""
        if len(self.focus_history) < 2:
            return
        
        x, y = position
        history = self.focus_history[-60:]  # Last 2 seconds at 30fps
        
        # Background
        cv2.rectangle(image, (x, y), (x + width, y + height), (30, 30, 30), -1)
        cv2.rectangle(image, (x, y), (x + width, y + height), (255, 255, 255), 1)
        
        # Draw score line
        if len(history) > 1:
            points = []
            for i, entry in enumerate(history):
                px = x + int((i / (len(history) - 1)) * width)
                py = y + height - int(entry['score'] * height)
                points.append((px, py))
            
            # Draw line segments with colors based on score
            for i in range(len(points) - 1):
                score = history[i]['score']
                color = self.get_focus_color(score)
                cv2.line(image, points[i], points[i + 1], color, 2)
        
        # Labels
        cv2.putText(image, "Focus History", (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(image, "1.0", (x - 20, y + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        cv2.putText(image, "0", (x - 15, y + height),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
    
    def cleanup(self):
        """Cleanup system actions on shutdown"""
        if self.screen_dimmed:
            self.restore_screen_brightness()
        print("⚙️ System actions cleaned up")