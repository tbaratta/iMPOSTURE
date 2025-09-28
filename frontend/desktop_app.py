"""
StraightUp Modern Desktop App
Beautiful desktop interface matching the web UI design
Project: perfect-entry-473503-j1
"""

import tkinter as tk
from tkinter import ttk, messagebox, font
import customtkinter as ctk
import threading
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

# Set Google Cloud Project
os.environ['GOOGLE_CLOUD_PROJECT'] = 'perfect-entry-473503-j1'

# Google Cloud logging
try:
    from google.cloud import logging as cloud_logging
    CLOUD_LOGGING_AVAILABLE = True
    print("✅ Google Cloud logging imports successful")
except ImportError:
    CLOUD_LOGGING_AVAILABLE = False
    print("⚠️ Google Cloud logging not available")

# Set modern appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class HealthDataManager:
    """Manages health data retrieval from Google Cloud"""
    
    def __init__(self):
        self.cloud_logging_client = None
        self.logger_name = 'straightup-adk-production'
        
        if CLOUD_LOGGING_AVAILABLE:
            try:
                self.cloud_logging_client = cloud_logging.Client(project='perfect-entry-473503-j1')
                print("🌐 Google Cloud logging client initialized")
            except Exception as e:
                print(f"⚠️ Cloud logging initialization failed: {e}")
    
    def get_recent_health_data(self, hours: int = 24, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch recent health data from Google Cloud Logging"""
        if not self.cloud_logging_client:
            return self._generate_sample_data()
        
        try:
            # Calculate time range
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours)
            
            # Query Google Cloud Logging
            filter_str = f'''
                logName="projects/perfect-entry-473503-j1/logs/{self.logger_name}"
                AND timestamp >= "{start_time.isoformat()}Z"
                AND timestamp <= "{end_time.isoformat()}Z"
                AND jsonPayload.source="adk_production_system"
            '''
            
            entries = list(self.cloud_logging_client.list_entries(
                filter_=filter_str,
                order_by=cloud_logging.DESCENDING,
                max_results=limit
            ))
            
            health_data = []
            for entry in entries:
                if hasattr(entry, 'payload') and isinstance(entry.payload, dict):
                    data_point = {
                        'timestamp': entry.timestamp.isoformat() if entry.timestamp else datetime.utcnow().isoformat(),
                        'focus_score': entry.payload.get('focus_score', 0.5),
                        'posture_score': entry.payload.get('posture_score', 0.5),
                        'phone_usage_seconds': entry.payload.get('phone_usage_seconds', 0.0),
                        'noise_level': entry.payload.get('noise_level', 0.3),
                        'recommendations': entry.payload.get('recommendations', []),
                        'cycle': entry.payload.get('cycle', 0),
                        'agent_status': entry.payload.get('agent_status', 'unknown')
                    }
                    health_data.append(data_point)
            
            print(f"📊 Retrieved {len(health_data)} health data points from Google Cloud")
            return health_data
            
        except Exception as e:
            print(f"⚠️ Error fetching from Google Cloud: {e}")
            return self._generate_sample_data()
    
    def get_health_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get aggregated health summary statistics"""
        data = self.get_recent_health_data(hours)
        
        if not data:
            return {
                'status': 'no_data',
                'message': 'No health data available'
            }
        
        # Calculate averages and trends
        total_points = len(data)
        avg_focus = sum(d['focus_score'] for d in data) / total_points
        avg_posture = sum(d['posture_score'] for d in data) / total_points
        total_phone_time = sum(d['phone_usage_seconds'] for d in data)
        avg_noise = sum(d['noise_level'] for d in data) / total_points
        
        # Get recent vs older data for trends
        recent_data = data[:total_points//3] if total_points > 6 else data
        older_data = data[total_points//3:] if total_points > 6 else data
        
        recent_focus = sum(d['focus_score'] for d in recent_data) / len(recent_data) if recent_data else avg_focus
        older_focus = sum(d['focus_score'] for d in older_data) / len(older_data) if older_data else avg_focus
        
        focus_trend = 'improving' if recent_focus > older_focus else 'declining' if recent_focus < older_focus else 'stable'
        
        # Collect all recommendations
        all_recommendations = []
        for entry in data:
            all_recommendations.extend(entry.get('recommendations', []))
        
        # Count recommendation frequency
        rec_counts = {}
        for rec in all_recommendations:
            rec_counts[rec] = rec_counts.get(rec, 0) + 1
        
        top_recommendations = sorted(rec_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Generate live data format from most recent entry
        live_data = self._generate_live_data_from_recent(data)
        
        return {
            'status': 'success',
            'data_points': total_points,
            'time_range_hours': hours,
            'averages': {
                'focus_score': round(avg_focus, 3),
                'posture_score': round(avg_posture, 3),
                'noise_level': round(avg_noise, 3)
            },
            'totals': {
                'phone_usage_seconds': round(total_phone_time, 1),
                'phone_usage_minutes': round(total_phone_time / 60, 1)
            },
            'trends': {
                'focus_trend': focus_trend,
                'recent_focus': round(recent_focus, 3),
                'older_focus': round(older_focus, 3)
            },
            'top_recommendations': [{'text': rec, 'count': count} for rec, count in top_recommendations],
            'health_grade': self._calculate_health_grade(avg_focus, avg_posture, total_phone_time, avg_noise),
            'last_updated': data[0]['timestamp'] if data else datetime.utcnow().isoformat(),
            'live_data': live_data,
            'metrics': {
                'distraction_level': round((1 - avg_focus) * 100, 1),
                'focus_score': round(avg_focus * 100, 1),
                'posture_score': round(avg_posture * 100, 1)
            }
        }
    
    def _generate_live_data_from_recent(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate live data format from recent entries"""
        if not data:
            return {
                'postureScore': {'value': '75%', 'status': 'good'},
                'phoneUsage': {'value': '2.3 min', 'status': 'ok'},
                'noiseLevel': {'value': '28%', 'status': 'good'},
                'focusScore': {'value': '64%', 'status': 'warn'}
            }
        
        # Use most recent 5 entries for live data
        recent_entries = data[:5]
        
        avg_posture = sum(d['posture_score'] for d in recent_entries) / len(recent_entries)
        avg_focus = sum(d['focus_score'] for d in recent_entries) / len(recent_entries)
        avg_noise = sum(d['noise_level'] for d in recent_entries) / len(recent_entries)
        total_phone = sum(d['phone_usage_seconds'] for d in recent_entries)
        
        return {
            'postureScore': {
                'value': f"{int(avg_posture * 100)}%",
                'status': 'good' if avg_posture > 0.7 else 'warn' if avg_posture > 0.4 else 'bad'
            },
            'focusScore': {
                'value': f"{int(avg_focus * 100)}%",
                'status': 'good' if avg_focus > 0.7 else 'warn' if avg_focus > 0.4 else 'bad'
            },
            'noiseLevel': {
                'value': f"{int(avg_noise * 100)}%",
                'status': 'good' if avg_noise < 0.3 else 'warn' if avg_noise < 0.6 else 'bad'
            },
            'phoneUsage': {
                'value': f"{total_phone / 60:.1f} min",
                'status': 'good' if total_phone < 300 else 'warn' if total_phone < 900 else 'bad'
            }
        }
    
    def _calculate_health_grade(self, focus: float, posture: float, phone_time: float, noise: float) -> str:
        """Calculate overall health grade"""
        focus_score = focus * 100
        posture_score = posture * 100
        phone_penalty = min(phone_time / 300 * 20, 20)
        noise_penalty = noise * 30
        
        overall = (focus_score + posture_score - phone_penalty - noise_penalty) / 2
        
        if overall >= 85:
            return 'A'
        elif overall >= 75:
            return 'B'
        elif overall >= 65:
            return 'C'
        elif overall >= 55:
            return 'D'
        else:
            return 'F'
    
    def _generate_sample_data(self) -> List[Dict[str, Any]]:
        """Generate sample data for demo purposes"""
        import random
        
        sample_data = []
        base_time = datetime.utcnow()
        
        for i in range(20):
            timestamp = base_time - timedelta(minutes=i * 5)
            sample_data.append({
                'timestamp': timestamp.isoformat(),
                'focus_score': round(random.uniform(0.4, 0.9), 3),
                'posture_score': round(random.uniform(0.3, 0.9), 3),
                'phone_usage_seconds': round(random.uniform(0, 45), 1),
                'noise_level': round(random.uniform(0.1, 0.6), 3),
                'recommendations': random.choice([
                    ['🎯 Focus on alignment', '✅ Good posture'],
                    ['🔴 Neck flexion critical - Raise monitor', '🟡 Minor shoulder imbalance'],
                    ['📱 Brief phone check', '🌟 Excellent posture!'],
                    []
                ]),
                'cycle': 100 - i,
                'agent_status': 'operational'
            })
        
        return sample_data

class ModernDesktopApp:
    """Modern desktop app matching the web UI design"""
    
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Straight Up — Desktop Dashboard")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Color scheme matching web UI
        self.colors = {
            'bg': '#0B0F17',
            'card': '#111827',
            'muted': '#9AA4B2',
            'border': '#273047',
            'ink': '#ffffff',
            'accent': '#4F46E5',
            'accent2': '#22C55E',
            'danger': '#ef4444',
            'warn': '#f59e0b',
            'panel': '#0F1626'
        }
        
        # Initialize data manager
        self.data_manager = HealthDataManager()
        
        # App state
        self.session_running = False
        self.session_paused = False
        self.session_start_time = None
        self.session_elapsed = 0
        self.timer_id = None
        self.auto_refresh = True
        self.current_data = None
        
        # Setup UI
        self.setup_ui()
        self.setup_refresh_timer()
        
        # Load initial data
        self.refresh_data()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Configure root window
        self.root.configure(fg_color=self.colors['bg'])
        
        # Main container with padding
        self.main_frame = ctk.CTkFrame(self.root, fg_color=self.colors['bg'])
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Setup components
        self.setup_header()
        self.setup_badges()
        self.setup_main_content()
    
    def setup_header(self):
        """Setup header with logo and controls"""
        header_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        header_frame.pack(fill="x", pady=(0, 16))
        
        # Brand section
        brand_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        brand_frame.pack(side="left", fill="y")
        
        # Logo (simplified as text since we can't easily add SVG)
        logo_frame = ctk.CTkFrame(
            brand_frame,
            width=36, height=36,
            fg_color=self.colors['accent'],
            corner_radius=10
        )
        logo_frame.pack(side="left", padx=(0, 12))
        logo_frame.pack_propagate(False)
        
        logo_label = ctk.CTkLabel(
            logo_frame,
            text="⬆",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="white"
        )
        logo_label.pack(expand=True)
        
        # Title
        title_label = ctk.CTkLabel(
            brand_frame,
            text="Straight Up",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=self.colors['ink']
        )
        title_label.pack(side="left")
        
        # Control buttons
        controls_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        controls_frame.pack(side="right")
        
        # Status indicator
        self.status_label = ctk.CTkLabel(
            controls_frame,
            text="🔴 Connecting...",
            font=ctk.CTkFont(size=12),
            text_color=self.colors['muted']
        )
        self.status_label.pack(side="right", padx=(0, 20))
        
        settings_btn = ctk.CTkButton(
            controls_frame,
            text="Settings",
            width=80,
            height=32,
            fg_color="transparent",
            border_width=1,
            border_color=self.colors['border'],
            text_color=self.colors['ink'],
            hover_color=self.colors['panel'],
            command=self.show_settings
        )
        settings_btn.pack(side="right", padx=5)
        
        refresh_btn = ctk.CTkButton(
            controls_frame,
            text="Refresh",
            width=80,
            height=32,
            fg_color=self.colors['danger'],
            hover_color="#dc2626",
            command=self.refresh_data
        )
        refresh_btn.pack(side="right", padx=5)
    
    def setup_badges(self):
        """Setup incentive badges"""
        badges_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        badges_frame.pack(fill="x", pady=(0, 16))
        
        # Level badge
        level_badge = self.create_badge(
            badges_frame,
            "🏅",
            "Lv. 1 — Starter",
            "Next: 80 min total",
            gradient_colors=[self.colors['accent2'], self.colors['accent']]
        )
        level_badge.pack(side="left", fill="x", expand=True, padx=(0, 6))
        
        # Today badge
        self.today_badge = self.create_badge(
            badges_frame,
            "⏱️",
            "0 min today",
            "Consistency beats intensity",
            gradient_colors=[self.colors['accent'], self.colors['accent2']]
        )
        self.today_badge.pack(side="left", fill="x", expand=True, padx=6)
        
        # Goal badge
        goal_badge = self.create_badge(
            badges_frame,
            "🎯",
            "Daily goal: 45 min",
            "Adjust in setup",
            gradient_colors=[self.colors['warn'], self.colors['accent']]
        )
        goal_badge.pack(side="left", fill="x", expand=True, padx=(6, 0))
    
    def create_badge(self, parent, icon, title, subtitle, gradient_colors):
        """Create a badge with icon and text"""
        badge_frame = ctk.CTkFrame(
            parent,
            fg_color=self.colors['card'],
            border_width=1,
            border_color=self.colors['border'],
            corner_radius=14
        )
        
        content_frame = ctk.CTkFrame(badge_frame, fg_color="transparent")
        content_frame.pack(fill="both", expand=True, padx=14, pady=14)
        
        # Icon
        icon_frame = ctk.CTkFrame(
            content_frame,
            width=36, height=36,
            fg_color=gradient_colors[0],
            corner_radius=10
        )
        icon_frame.pack(side="left", padx=(0, 10))
        icon_frame.pack_propagate(False)
        
        icon_label = ctk.CTkLabel(
            icon_frame,
            text=icon,
            font=ctk.CTkFont(size=16)
        )
        icon_label.pack(expand=True)
        
        # Text content
        text_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
        text_frame.pack(side="left", fill="both", expand=True)
        
        title_label = ctk.CTkLabel(
            text_frame,
            text=title,
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=self.colors['ink'],
            anchor="w"
        )
        title_label.pack(fill="x")
        
        subtitle_label = ctk.CTkLabel(
            text_frame,
            text=subtitle,
            font=ctk.CTkFont(size=10),
            text_color=self.colors['muted'],
            anchor="w"
        )
        subtitle_label.pack(fill="x")
        
        return badge_frame
    
    def setup_main_content(self):
        """Setup main two-column content area"""
        content_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        content_frame.pack(fill="both", expand=True)
        
        # Session card (left column)
        self.setup_session_card(content_frame)
        
        # Wellness card (right column)
        self.setup_wellness_card(content_frame)
    
    def setup_session_card(self, parent):
        """Setup session control card"""
        session_card = ctk.CTkFrame(
            parent,
            fg_color=self.colors['card'],
            border_width=1,
            border_color=self.colors['border'],
            corner_radius=16
        )
        session_card.pack(side="left", fill="both", expand=True, padx=(0, 8))
        
        # Card content
        card_content = ctk.CTkFrame(session_card, fg_color="transparent")
        card_content.pack(fill="both", expand=True, padx=18, pady=18)
        
        # Welcome message
        self.welcome_label = ctk.CTkLabel(
            card_content,
            text="Welcome back",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=self.colors['ink'],
            anchor="w"
        )
        self.welcome_label.pack(fill="x", pady=(0, 6))
        
        # Session title
        session_title = ctk.CTkLabel(
            card_content,
            text="Session",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=self.colors['ink'],
            anchor="w"
        )
        session_title.pack(fill="x", pady=(0, 10))
        
        # Session subtitle
        self.session_subtitle = ctk.CTkLabel(
            card_content,
            text="Configure your session, then start. We'll time it and track progress.",
            font=ctk.CTkFont(size=12),
            text_color=self.colors['muted'],
            anchor="w",
            wraplength=400
        )
        self.session_subtitle.pack(fill="x", pady=(0, 8))
        
        # Timer display
        timer_frame = ctk.CTkFrame(
            card_content,
            fg_color=self.colors['panel'],
            border_width=1,
            border_color="#2a3350",
            corner_radius=14
        )
        timer_frame.pack(fill="x", pady=(0, 10))
        
        self.timer_label = ctk.CTkLabel(
            timer_frame,
            text="00:00",
            font=ctk.CTkFont(size=48, weight="bold"),
            text_color=self.colors['ink']
        )
        self.timer_label.pack(pady=18)
        
        # Control buttons
        controls_frame = ctk.CTkFrame(card_content, fg_color="transparent")
        controls_frame.pack(fill="x", pady=(0, 10))
        
        self.setup_btn = ctk.CTkButton(
            controls_frame,
            text="Open session setup",
            fg_color="transparent",
            border_width=1,
            border_color=self.colors['border'],
            text_color=self.colors['ink'],
            hover_color=self.colors['panel'],
            command=self.open_session_setup
        )
        self.setup_btn.pack(side="left", padx=(0, 5))
        
        self.pause_btn = ctk.CTkButton(
            controls_frame,
            text="Pause",
            fg_color="transparent",
            border_width=1,
            border_color=self.colors['border'],
            text_color=self.colors['ink'],
            hover_color=self.colors['panel'],
            state="disabled",
            command=self.pause_session
        )
        self.pause_btn.pack(side="left", padx=5)
        
        self.stop_btn = ctk.CTkButton(
            controls_frame,
            text="Stop & Save",
            fg_color="transparent",
            border_width=1,
            border_color=self.colors['border'],
            text_color=self.colors['ink'],
            hover_color=self.colors['panel'],
            state="disabled",
            command=self.stop_session
        )
        self.stop_btn.pack(side="left", padx=5)
        
        # Status pill
        status_frame = ctk.CTkFrame(
            controls_frame,
            fg_color=self.colors['panel'],
            border_width=1,
            border_color=self.colors['border'],
            corner_radius=999
        )
        status_frame.pack(side="right")
        
        status_content = ctk.CTkFrame(status_frame, fg_color="transparent")
        status_content.pack(padx=14, pady=10)
        
        # Status dot (simulated with colored text)
        self.status_dot = ctk.CTkLabel(
            status_content,
            text="●",
            font=ctk.CTkFont(size=12),
            text_color=self.colors['accent2']
        )
        self.status_dot.pack(side="left", padx=(0, 6))
        
        self.session_status_label = ctk.CTkLabel(
            status_content,
            text="Idle",
            font=ctk.CTkFont(size=12),
            text_color=self.colors['ink']
        )
        self.session_status_label.pack(side="left")
        
        # Live summary section
        self.setup_live_summary(card_content)
    
    def setup_live_summary(self, parent):
        """Setup live wellness summary"""
        summary_frame = ctk.CTkFrame(
            parent,
            fg_color=self.colors['card'],
            border_width=1,
            border_color=self.colors['border'],
            corner_radius=16
        )
        summary_frame.pack(fill="x", pady=(18, 0))
        
        # Header
        header_frame = ctk.CTkFrame(summary_frame, fg_color="transparent")
        header_frame.pack(fill="x", padx=16, pady=(16, 10))
        
        summary_title = ctk.CTkLabel(
            header_frame,
            text="Live wellness",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=self.colors['ink']
        )
        summary_title.pack(side="left")
        
        self.live_dot = ctk.CTkLabel(
            header_frame,
            text="●",
            font=ctk.CTkFont(size=10),
            text_color=self.colors['muted']
        )
        self.live_dot.pack(side="right")
        
        # Summary content
        self.live_summary_frame = ctk.CTkFrame(summary_frame, fg_color="transparent")
        self.live_summary_frame.pack(fill="x", padx=16, pady=(0, 16))
        
        # Initial loading chip
        loading_chip = self.create_chip(self.live_summary_frame, "Loading live summary…", "muted")
        loading_chip.pack(side="left")
    
    def create_chip(self, parent, text, status="muted", icon="•"):
        """Create a status chip"""
        colors = {
            'good': {'fg': self.colors['accent2'], 'border': self.colors['accent2']},
            'warn': {'fg': self.colors['warn'], 'border': self.colors['warn']},
            'bad': {'fg': self.colors['danger'], 'border': self.colors['danger']},
            'muted': {'fg': self.colors['muted'], 'border': self.colors['border']}
        }
        
        chip_colors = colors.get(status, colors['muted'])
        
        chip = ctk.CTkFrame(
            parent,
            fg_color="transparent",
            border_width=1,
            border_color=chip_colors['border'],
            corner_radius=999
        )
        
        content_frame = ctk.CTkFrame(chip, fg_color="transparent")
        content_frame.pack(padx=14, pady=10)
        
        # Icon
        icon_label = ctk.CTkLabel(
            content_frame,
            text=icon,
            font=ctk.CTkFont(size=14),
            text_color=chip_colors['fg']
        )
        icon_label.pack(side="left", padx=(0, 8))
        
        # Text
        text_label = ctk.CTkLabel(
            content_frame,
            text=text,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=chip_colors['fg']
        )
        text_label.pack(side="left")
        
        return chip
    
    def setup_wellness_card(self, parent):
        """Setup wellness report card"""
        wellness_card = ctk.CTkFrame(
            parent,
            fg_color=self.colors['card'],
            border_width=1,
            border_color=self.colors['border'],
            corner_radius=16
        )
        wellness_card.pack(side="right", fill="both", expand=True, padx=(8, 0))
        
        # Card content
        card_content = ctk.CTkFrame(wellness_card, fg_color="transparent")
        card_content.pack(fill="both", expand=True, padx=18, pady=18)
        
        # Title
        title_label = ctk.CTkLabel(
            card_content,
            text="Wellness report",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=self.colors['ink'],
            anchor="w"
        )
        title_label.pack(fill="x", pady=(0, 10))
        
        # Subtitle
        subtitle_label = ctk.CTkLabel(
            card_content,
            text="Snapshot of posture, focus, distractions, and break habits.",
            font=ctk.CTkFont(size=12),
            text_color=self.colors['muted'],
            anchor="w",
            wraplength=400
        )
        subtitle_label.pack(fill="x", pady=(0, 8))
        
        # Metrics grid
        metrics_frame = ctk.CTkFrame(card_content, fg_color="transparent")
        metrics_frame.pack(fill="both", expand=True)
        
        # Configure grid
        metrics_frame.grid_columnconfigure(0, weight=1)
        metrics_frame.grid_columnconfigure(1, weight=1)
        metrics_frame.grid_rowconfigure(0, weight=1)
        metrics_frame.grid_rowconfigure(1, weight=1)
        
        # Create metric widgets
        self.distraction_metric = self.create_metric_widget(
            metrics_frame, "Distraction level", "Lower is better."
        )
        self.distraction_metric.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=(0, 6))
        
        self.focus_metric = self.create_metric_widget(
            metrics_frame, "Focus score", "Sustained work time."
        )
        self.focus_metric.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=(0, 6))
        
        self.posture_metric = self.create_metric_widget(
            metrics_frame, "Posture score", "Neck and shoulder alignment."
        )
        self.posture_metric.grid(row=1, column=0, sticky="nsew", padx=(0, 6), pady=(6, 0))
        
        # Empty space for symmetry
        empty_frame = ctk.CTkFrame(metrics_frame, fg_color="transparent")
        empty_frame.grid(row=1, column=1, sticky="nsew", padx=(6, 0), pady=(6, 0))
    
    def create_metric_widget(self, parent, title, hint):
        """Create a metric display widget with progress bar"""
        metric_frame = ctk.CTkFrame(
            parent,
            fg_color=self.colors['panel'],
            border_width=1,
            border_color=self.colors['border'],
            corner_radius=12
        )
        
        content_frame = ctk.CTkFrame(metric_frame, fg_color="transparent")
        content_frame.pack(fill="both", expand=True, padx=14, pady=14)
        
        # Header with title and percentage
        header_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
        header_frame.pack(fill="x", pady=(0, 6))
        
        title_label = ctk.CTkLabel(
            header_frame,
            text=title,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=self.colors['ink'],
            anchor="w"
        )
        title_label.pack(side="left")
        
        pct_label = ctk.CTkLabel(
            header_frame,
            text="—",
            font=ctk.CTkFont(size=12),
            text_color=self.colors['muted']
        )
        pct_label.pack(side="right")
        
        # Progress bar
        progress_bg = ctk.CTkFrame(
            content_frame,
            height=12,
            fg_color="#0c1220",
            border_width=1,
            border_color="#223055",
            corner_radius=999
        )
        progress_bg.pack(fill="x", pady=(0, 6))
        progress_bg.pack_propagate(False)
        
        progress_bar = ctk.CTkProgressBar(
            progress_bg,
            height=10,
            progress_color=self.colors['accent2']
        )
        progress_bar.pack(fill="both", expand=True, padx=1, pady=1)
        progress_bar.set(0)
        
        # Hint
        hint_label = ctk.CTkLabel(
            content_frame,
            text=hint,
            font=ctk.CTkFont(size=10),
            text_color=self.colors['muted'],
            anchor="w"
        )
        hint_label.pack(fill="x")
        
        # Store references for updates
        metric_frame.pct_label = pct_label
        metric_frame.progress_bar = progress_bar
        
        return metric_frame
    
    def setup_refresh_timer(self):
        """Setup auto-refresh timer"""
        if self.auto_refresh:
            self.root.after(5000, self.auto_refresh_data)  # Refresh every 5 seconds
    
    def auto_refresh_data(self):
        """Auto-refresh data if enabled"""
        if self.auto_refresh:
            self.refresh_data()
            self.setup_refresh_timer()
    
    def refresh_data(self):
        """Refresh health data from backend"""
        def fetch_data():
            try:
                self.current_data = self.data_manager.get_health_summary()
                self.root.after(0, self.update_ui)
            except Exception as e:
                self.root.after(0, lambda: self.show_error(f"Failed to fetch data: {e}"))
        
        # Run in background thread
        thread = threading.Thread(target=fetch_data, daemon=True)
        thread.start()
        
        # Update status
        self.status_label.configure(text="🔄 Refreshing...")
    
    def update_ui(self):
        """Update UI with current data"""
        if not self.current_data or self.current_data.get('status') != 'success':
            self.status_label.configure(text="🔴 No Data Available")
            return
        
        data = self.current_data
        
        # Update status
        self.status_label.configure(text="🟢 System Active")
        
        # Update metrics
        metrics = data.get('metrics', {})
        
        # Distraction level (inverted for display)
        dist_pct = metrics.get('distraction_level', 28)
        self.distraction_metric.pct_label.configure(text=f"{100-dist_pct}%")
        self.distraction_metric.progress_bar.set((100-dist_pct) / 100)
        
        # Focus score
        focus_pct = metrics.get('focus_score', 64)
        self.focus_metric.pct_label.configure(text=f"{focus_pct}%")
        self.focus_metric.progress_bar.set(focus_pct / 100)
        
        # Posture score
        posture_pct = metrics.get('posture_score', 75)
        self.posture_metric.pct_label.configure(text=f"{posture_pct}%")
        self.posture_metric.progress_bar.set(posture_pct / 100)
        
        # Update live summary
        self.update_live_summary(data.get('live_data', {}))
    
    def update_live_summary(self, live_data):
        """Update live wellness summary chips"""
        # Clear existing chips
        for widget in self.live_summary_frame.winfo_children():
            widget.destroy()
        
        if not live_data:
            loading_chip = self.create_chip(self.live_summary_frame, "No live data available", "muted")
            loading_chip.pack(side="left")
            return
        
        # Create chips for each metric
        chip_count = 0
        max_chips = 3  # Limit to prevent overflow
        
        for key, value in live_data.items():
            if chip_count >= max_chips:
                break
                
            # Extract status and value
            if isinstance(value, dict):
                status = value.get('status', 'muted')
                display_value = value.get('value', '—')
            else:
                status = 'muted'
                display_value = str(value)
            
            # Create friendly names and icons
            display_name = self.get_friendly_name(key)
            icon = self.get_metric_icon(key)
            
            # Determine chip status
            chip_status = self.get_chip_status(status)
            
            chip_text = f"{display_name}: {display_value}"
            chip = self.create_chip(self.live_summary_frame, chip_text, chip_status, icon)
            chip.pack(side="left", padx=(0, 12))
            
            chip_count += 1
        
        # Update live dot color
        if chip_count > 0:
            self.live_dot.configure(text_color=self.colors['accent2'])
        else:
            self.live_dot.configure(text_color=self.colors['muted'])
    
    def get_friendly_name(self, key):
        """Convert API key to friendly display name"""
        names = {
            'postureScore': 'Posture',
            'phoneUsage': 'Phone',
            'noiseLevel': 'Noise',
            'focusScore': 'Focus'
        }
        return names.get(key, key)
    
    def get_metric_icon(self, key):
        """Get icon for metric"""
        icons = {
            'postureScore': '🧍‍♂️',
            'phoneUsage': '📱',
            'noiseLevel': '🔊',
            'focusScore': '🎯'
        }
        return icons.get(key, '•')
    
    def get_chip_status(self, status):
        """Convert API status to chip status"""
        status_map = {
            'good': 'good',
            'ok': 'good',
            'warn': 'warn',
            'warning': 'warn',
            'bad': 'bad',
            'error': 'bad'
        }
        return status_map.get(status.lower(), 'muted')
    
    def open_session_setup(self):
        """Open session setup dialog"""
        setup_window = ctk.CTkToplevel(self.root)
        setup_window.title("Session Setup")
        setup_window.geometry("400x500")
        setup_window.configure(fg_color=self.colors['card'])
        
        # Setup window content
        content_frame = ctk.CTkFrame(setup_window, fg_color="transparent")
        content_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        title_label = ctk.CTkLabel(
            content_frame,
            text="Session setup",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=self.colors['ink']
        )
        title_label.pack(pady=(0, 20))
        
        # Camera preview placeholder
        camera_frame = ctk.CTkFrame(
            content_frame,
            height=200,
            fg_color=self.colors['panel'],
            border_width=1,
            border_color=self.colors['border'],
            corner_radius=12
        )
        camera_frame.pack(fill="x", pady=(0, 10))
        camera_frame.pack_propagate(False)
        
        camera_label = ctk.CTkLabel(
            camera_frame,
            text="📹 Webcam Preview\n(used for posture & distraction checks)",
            font=ctk.CTkFont(size=12),
            text_color=self.colors['muted']
        )
        camera_label.pack(expand=True)
        
        # Settings
        settings_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
        settings_frame.pack(fill="x", pady=(0, 20))
        
        # Break frequency
        break_freq_label = ctk.CTkLabel(
            settings_frame,
            text="Break frequency (minutes)",
            font=ctk.CTkFont(size=12),
            text_color=self.colors['ink'],
            anchor="w"
        )
        break_freq_label.pack(fill="x", pady=(0, 5))
        
        break_freq_combo = ctk.CTkComboBox(
            settings_frame,
            values=["25", "45", "50", "60"],
            state="readonly"
        )
        break_freq_combo.set("45")
        break_freq_combo.pack(fill="x", pady=(0, 10))
        
        # Break length
        break_len_label = ctk.CTkLabel(
            settings_frame,
            text="Break length (minutes)",
            font=ctk.CTkFont(size=12),
            text_color=self.colors['ink'],
            anchor="w"
        )
        break_len_label.pack(fill="x", pady=(0, 5))
        
        break_len_combo = ctk.CTkComboBox(
            settings_frame,
            values=["3", "5", "10"],
            state="readonly"
        )
        break_len_combo.set("5")
        break_len_combo.pack(fill="x", pady=(0, 10))
        
        # Distraction tracking
        distraction_label = ctk.CTkLabel(
            settings_frame,
            text="Distraction tracking",
            font=ctk.CTkFont(size=12),
            text_color=self.colors['ink'],
            anchor="w"
        )
        distraction_label.pack(fill="x", pady=(0, 5))
        
        distraction_switch = ctk.CTkSwitch(
            settings_frame,
            text="Alert for prolonged phone usage"
        )
        distraction_switch.select()
        distraction_switch.pack(fill="x", pady=(0, 20))
        
        # Buttons
        button_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
        button_frame.pack(fill="x")
        
        start_btn = ctk.CTkButton(
            button_frame,
            text="Start session",
            command=lambda: self.start_session(setup_window)
        )
        start_btn.pack(side="left", padx=(0, 10))
        
        close_btn = ctk.CTkButton(
            button_frame,
            text="Close",
            fg_color="transparent",
            border_width=1,
            border_color=self.colors['border'],
            text_color=self.colors['ink'],
            command=setup_window.destroy
        )
        close_btn.pack(side="left")
    
    def start_session(self, setup_window=None):
        """Start a monitoring session"""
        if setup_window:
            setup_window.destroy()
        
        self.session_running = True
        self.session_paused = False
        self.session_start_time = datetime.now()
        self.session_elapsed = 0
        
        # Update UI
        self.session_subtitle.configure(text="Timer active. We're tracking posture and focus.")
        self.pause_btn.configure(state="normal")
        self.stop_btn.configure(state="normal")
        self.session_status_label.configure(text="Running")
        self.status_dot.configure(text_color=self.colors['danger'])
        
        # Start timer
        self.update_timer()
    
    def pause_session(self):
        """Pause/resume the session"""
        if not self.session_running:
            return
        
        self.session_paused = not self.session_paused
        
        if self.session_paused:
            self.session_elapsed += (datetime.now() - self.session_start_time).total_seconds()
            self.session_subtitle.configure(text="Resume when you're ready.")
            self.pause_btn.configure(text="Resume")
            self.session_status_label.configure(text="Paused")
            self.status_dot.configure(text_color=self.colors['warn'])
        else:
            self.session_start_time = datetime.now()
            self.session_subtitle.configure(text="Timer active. We're tracking posture and focus.")
            self.pause_btn.configure(text="Pause")
            self.session_status_label.configure(text="Running")
            self.status_dot.configure(text_color=self.colors['danger'])
        
        self.update_timer()
    
    def stop_session(self):
        """Stop and save the session"""
        if not self.session_running:
            return
        
        # Calculate total time
        if not self.session_paused:
            self.session_elapsed += (datetime.now() - self.session_start_time).total_seconds()
        
        # Reset session state
        self.session_running = False
        self.session_paused = False
        
        # Update UI
        self.session_subtitle.configure(text="Configure your session, then start. We'll time it and track progress.")
        self.pause_btn.configure(state="disabled", text="Pause")
        self.stop_btn.configure(state="disabled")
        self.session_status_label.configure(text="Idle")
        self.status_dot.configure(text_color=self.colors['accent2'])
        self.timer_label.configure(text="00:00")
        
        # Show session summary
        minutes = int(self.session_elapsed // 60)
        seconds = int(self.session_elapsed % 60)
        
        if minutes > 0:
            messagebox.showinfo(
                "Session Complete",
                f"Session saved! Duration: {minutes}m {seconds}s\n\nGreat work on your focus session!"
            )
            
            # Update today badge
            # This would normally save to backend/database
            self.today_badge.winfo_children()[0].winfo_children()[1].winfo_children()[0].configure(
                text=f"{minutes} min today"
            )
        else:
            messagebox.showwarning(
                "Session Too Short",
                "Session was too short to save (< 1 minute)."
            )
    
    def update_timer(self):
        """Update the session timer"""
        if self.session_running and not self.session_paused:
            current_elapsed = self.session_elapsed + (datetime.now() - self.session_start_time).total_seconds()
        else:
            current_elapsed = self.session_elapsed
        
        minutes = int(current_elapsed // 60)
        seconds = int(current_elapsed % 60)
        
        self.timer_label.configure(text=f"{minutes:02d}:{seconds:02d}")
        
        if self.session_running:
            self.timer_id = self.root.after(1000, self.update_timer)
    
    def show_settings(self):
        """Show settings dialog"""
        settings_window = ctk.CTkToplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("350x300")
        settings_window.configure(fg_color=self.colors['card'])
        
        content_frame = ctk.CTkFrame(settings_window, fg_color="transparent")
        content_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        title_label = ctk.CTkLabel(
            content_frame,
            text="Settings",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=self.colors['ink']
        )
        title_label.pack(pady=(0, 20))
        
        # Auto refresh setting
        refresh_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
        refresh_frame.pack(fill="x", pady=(0, 10))
        
        refresh_label = ctk.CTkLabel(
            refresh_frame,
            text="Auto refresh data",
            font=ctk.CTkFont(size=12),
            text_color=self.colors['ink']
        )
        refresh_label.pack(anchor="w")
        
        refresh_switch = ctk.CTkSwitch(
            refresh_frame,
            text="Enable automatic data refresh"
        )
        if self.auto_refresh:
            refresh_switch.select()
        refresh_switch.pack(anchor="w", pady=(5, 0))
        
        # Project info
        info_frame = ctk.CTkFrame(content_frame, fg_color=self.colors['panel'], corner_radius=10)
        info_frame.pack(fill="x", pady=(20, 0))
        
        info_content = ctk.CTkFrame(info_frame, fg_color="transparent")
        info_content.pack(fill="both", expand=True, padx=15, pady=15)
        
        info_text = f"""🎯 Project: perfect-entry-473503-j1
📊 Real-time ADK health monitoring
🌐 Google Cloud integration: {'✅' if CLOUD_LOGGING_AVAILABLE else '❌'}
🖥️ Desktop interface version"""
        
        info_label = ctk.CTkLabel(
            info_content,
            text=info_text,
            font=ctk.CTkFont(size=11),
            text_color=self.colors['muted'],
            anchor="w",
            justify="left"
        )
        info_label.pack(fill="both", expand=True)
        
        # Close button
        close_btn = ctk.CTkButton(
            content_frame,
            text="Close",
            command=settings_window.destroy
        )
        close_btn.pack(pady=(20, 0))
    
    def show_error(self, message: str):
        """Show error message"""
        messagebox.showerror("Error", message)
        self.status_label.configure(text="🔴 Error")
    
    def run(self):
        """Run the desktop application"""
        print("🚀 Starting Modern StraightUp Desktop App...")
        print("🎯 Project: perfect-entry-473503-j1")
        print("📊 Modern UI matching web design")
        
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("\n🛑 Desktop app closed by user")
        except Exception as e:
            print(f"❌ Desktop app error: {e}")

if __name__ == "__main__":
    print("🖥️ StraightUp Modern Desktop Dashboard")
    print("=" * 50)
    print("🎯 Project: perfect-entry-473503-j1")
    print("📊 Beautiful desktop interface")
    print("🎨 Matching web UI design")
    print("=" * 50)
    
    app = ModernDesktopApp()
    app.run()