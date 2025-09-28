"""
StraightUp - Frontend Dashboard Backend API
Pulls data from Google Cloud Logging and serves it to frontend dashboard
Project: perfect-entry-473503-j1
"""

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import os
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any, Optional

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

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

class StraightUpDashboardAPI:
    """Backend API for StraightUp Dashboard"""
    
    def __init__(self):
        self.cloud_logging_client = None
        self.logger_name = 'straightup-adk-production'  # Main production logger
        
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
                    # Convert Cloud Logging entry to our format
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
            'last_updated': data[0]['timestamp'] if data else datetime.utcnow().isoformat()
        }
    
    def _calculate_health_grade(self, focus: float, posture: float, phone_time: float, noise: float) -> str:
        """Calculate overall health grade"""
        # Normalize scores
        focus_score = focus * 100
        posture_score = posture * 100
        phone_penalty = min(phone_time / 300 * 20, 20)  # Max 20% penalty for 5+ minutes
        noise_penalty = noise * 30  # 0-30% penalty based on noise
        
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
        from datetime import timedelta
        
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

# Initialize API
dashboard_api = StraightUpDashboardAPI()

# API Routes
@app.route('/')
def index():
    """Serve the dashboard homepage"""
    return render_template('dashboard.html')

@app.route('/api/health/recent')
def get_recent_health():
    """Get recent health data"""
    hours = request.args.get('hours', 24, type=int)
    limit = request.args.get('limit', 100, type=int)
    
    data = dashboard_api.get_recent_health_data(hours, limit)
    return jsonify({
        'status': 'success',
        'data': data,
        'count': len(data)
    })

@app.route('/api/health/summary')
def get_health_summary():
    """Get health summary statistics"""
    hours = request.args.get('hours', 24, type=int)
    
    summary = dashboard_api.get_health_summary(hours)
    return jsonify(summary)

@app.route('/api/health/realtime')
def get_realtime_status():
    """Get current real-time status"""
    # Get the most recent data point
    recent_data = dashboard_api.get_recent_health_data(hours=1, limit=1)
    
    if not recent_data:
        return jsonify({
            'status': 'no_data',
            'message': 'No recent data available'
        })
    
    latest = recent_data[0]
    current_time = datetime.utcnow()
    last_update = datetime.fromisoformat(latest['timestamp'].replace('Z', '+00:00'))
    minutes_ago = (current_time.replace(tzinfo=None) - last_update.replace(tzinfo=None)).total_seconds() / 60
    
    return jsonify({
        'status': 'success',
        'current_metrics': latest,
        'last_update_minutes_ago': round(minutes_ago, 1),
        'is_recent': minutes_ago < 5,  # Consider data recent if less than 5 minutes old
        'system_status': 'active' if minutes_ago < 5 else 'idle'
    })

@app.route('/api/charts/focus-trend')
def get_focus_trend():
    """Get focus trend data for charts"""
    hours = request.args.get('hours', 12, type=int)
    data = dashboard_api.get_recent_health_data(hours)
    
    # Format for chart.js
    chart_data = {
        'labels': [datetime.fromisoformat(d['timestamp'].replace('Z', '')).strftime('%H:%M') for d in reversed(data)],
        'datasets': [
            {
                'label': 'Focus Score',
                'data': [d['focus_score'] for d in reversed(data)],
                'borderColor': 'rgb(75, 192, 192)',
                'backgroundColor': 'rgba(75, 192, 192, 0.2)',
                'tension': 0.1
            },
            {
                'label': 'Posture Score', 
                'data': [d['posture_score'] for d in reversed(data)],
                'borderColor': 'rgb(255, 99, 132)',
                'backgroundColor': 'rgba(255, 99, 132, 0.2)',
                'tension': 0.1
            }
        ]
    }
    
    return jsonify(chart_data)

if __name__ == '__main__':
    print("🌐 StraightUp Dashboard API Server")
    print("=" * 50)
    print(f"🎯 Project: perfect-entry-473503-j1")
    print("📊 Serving health data from Google Cloud Logging")
    print("🔗 Frontend dashboard available at: http://localhost:5000")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=5000, debug=True)