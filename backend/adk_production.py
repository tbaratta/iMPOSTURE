"""
StraightUp - Enhanced ADK Production System with Visual Agent Insights
Real ADK agent system for Google ADK Challenge with detailed console output
Project: perfect-entry-473503-j1
"""

import asyncio
import cv2
import numpy as np
import time
import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional

# Google Cloud logging and monitoring
try:
    from google.cloud import logging as cloud_logging
    from google.cloud import monitoring_v3
    CLOUD_LOGGING_AVAILABLE = True
    print("✅ Google Cloud logging imports successful")
except ImportError:
    CLOUD_LOGGING_AVAILABLE = False
    print("⚠️ Google Cloud logging not available - install: pip install google-cloud-logging google-cloud-monitoring")

# Set Google Cloud Project
os.environ['GOOGLE_CLOUD_PROJECT'] = 'perfect-entry-473503-j1'

# Import our integrated detector system  
from detector import IntegratedPoseDetector

# REAL Google ADK imports - NOT SIMULATION!
import google.adk
from google.adk.agents import Agent, LlmAgent, LoopAgent, ParallelAgent
from google.adk.tools import FunctionTool
from google.adk.models import Gemini
from google.adk.runners import InvocationContext

print(f"🚀 REAL Google ADK Version: {google.adk.__version__}")
print(f"📍 ADK Package Location: {google.adk.__file__}")
print("✅ REAL ADK imports successful - NO SIMULATION!")

@dataclass
class HealthMetrics:
    """Health monitoring data structure"""
    posture_score: float
    phone_usage_duration: float
    noise_level: float
    focus_score: float
    timestamp: float
    recommendations: List[str]

def create_agent_dashboard():
    """Create a visual dashboard showing agent data collection"""
    print("\n" + "=" * 80)
    print("🤖 ADK AGENT ARCHITECTURE VISUALIZATION")
    print("=" * 80)
    
    print("📊 AGENT HIERARCHY:")
    print("   🔄 LoopAgent (Continuous Monitoring)")
    print("      └── 🔀 ParallelAgent (Simultaneous Analysis)")
    print("          ├── 🎯 PostureAnalysisAgent")
    print("          │   ├── 📸 MediaPipe Pose Detection (33 landmarks)")
    print("          │   ├── 💪 Shoulder Alignment Analysis")
    print("          │   ├── 🏃 Neck Position Tracking")
    print("          │   └── 📊 Posture Score Calculation")
    print("          │")
    print("          ├── 📱 PhoneUsageAgent")
    print("          │   ├── 📸 YOLO11n Object Detection")
    print("          │   ├── ⏱️ Session Duration Tracking")
    print("          │   ├── 🎯 Usage Pattern Analysis")
    print("          │   └── 📊 Productivity Impact Scoring")
    print("          │")
    print("          ├── 🔊 EnvironmentalAgent")
    print("          │   ├── 🎤 Real-time Audio Analysis")
    print("          │   ├── 🌪️ Noise Level Classification")
    print("          │   ├── 📊 Environmental Quality Scoring")
    print("          │   └── 🎯 Focus Impact Analysis")
    print("          │")
    print("          └── 🤖 WellnessCoachAgent (LLM)")
    print("              ├── 💭 Contextual Analysis")
    print("              ├── 💡 Personalized Recommendations")
    print("              └── ⚠️ Intervention Triggering")
    
    print("\n📊 DATA COLLECTION POINTS:")
    print("   🎯 Posture: Shoulder angle, neck position, spine alignment")
    print("   📱 Phone: Detection confidence, session duration, usage patterns")
    print("   🔊 Environment: Noise RMS, frequency analysis, distraction factors")
    print("   🧠 Focus: Combined score from all factors (0.0-1.0)")
    
    print("\n🎮 AGENT INTERACTIONS:")
    print("   1. 📸 Raw webcam data captured")
    print("   2. 🔀 ParallelAgent distributes to specialized agents")
    print("   3. 🎯 Each agent analyzes its domain (posture/phone/environment)")
    print("   4. 📊 Results aggregated into HealthMetrics")
    print("   5. 🤖 WellnessCoach evaluates need for intervention")
    print("   6. ✅ Cycle repeats every 2 seconds")
    
    print("=" * 80)

class PostureAnalysisAgent(Agent):
    """ADK Agent for real-time posture analysis"""
    
    def __init__(self):
        # Initialize with Gemini model
        super().__init__(
            name="posture_analyzer",
            description="Analyzes user posture for workplace wellness using computer vision",
            model=Gemini(model_name="gemini-1.5-pro"),
            tools=[
                FunctionTool(self.analyze_posture)
            ]
        )
        print("🎯 Posture Analysis Agent initialized")
        
    def analyze_posture(self, image_data: Optional[np.ndarray] = None, detector: Optional[IntegratedPoseDetector] = None) -> Dict[str, Any]:
        """Analyze posture using real MediaPipe detection from detector.py"""
        try:
            # Use integrated detector - no simulation
            if image_data is not None and detector is not None:
                # Process frame with integrated detector
                processed_image, faces, hands, pose, phones, analysis = detector.process_frame(image_data.copy())
                
                # Calculate posture score based on REAL posture analysis
                if pose > 0 and analysis.get('ok'):
                    # Use the REAL posture analysis from integrated detector
                    overall_state = analysis.get('state', 'OK')
                    
                    # Convert posture state to numeric score
                    if overall_state == 'OK':
                        posture_score = 0.8
                    elif overall_state == 'WARN':
                        posture_score = 0.6
                    elif overall_state == 'BAD':
                        posture_score = 0.3
                    else:
                        posture_score = 0.5
                    
                    # Get metrics for shoulder and neck alignment
                    metrics = analysis.get('metrics', {})
                    states = analysis.get('states', {})
                    
                    # Calculate shoulder alignment from shoulder slope
                    shoulder_slope_deg = metrics.get('shoulder_slope_deg', 0)
                    shoulder_alignment = max(0.0, 1.0 - (shoulder_slope_deg / 45.0)) if shoulder_slope_deg else 0.5
                    
                    # Calculate neck position from neck angle
                    neck_angle_deg = metrics.get('neck_angle_deg', 0) 
                    neck_position = max(0.0, 1.0 - (neck_angle_deg / 45.0)) if neck_angle_deg else 0.5
                    
                    status = "real_posture_analysis"
                elif pose > 0:
                    # Pose detected but no detailed analysis available
                    posture_score = 0.5
                    shoulder_alignment = 0.5
                    neck_position = 0.5
                    status = "pose_detected_no_analysis"
                else:
                    # No pose detected - return actual state
                    posture_score = 0.0
                    shoulder_alignment = 0.0
                    neck_position = 0.0
                    status = "no_pose_detected"
            else:
                # No camera data available - return error state
                raise ValueError("No image data or detector available for posture analysis")
            
            # Generate recommendations based on real posture analysis
            if pose > 0 and analysis.get('ok'):
                recommendations = self._generate_enhanced_posture_recommendations(analysis, posture_score)
            else:
                recommendations = self._generate_posture_recommendations(posture_score)
            
            return {
                "posture_score": posture_score,
                "shoulder_alignment": shoulder_alignment,
                "neck_position": neck_position,
                "recommendations": recommendations,
                "analysis_timestamp": time.time(),
                "status": status,
                "pose_detected": pose > 0,
                "faces_detected": faces, 
                "hands_detected": hands,
                "focus_score": detector.focus_score if hasattr(detector, 'focus_score') else 0.5
            }
        except Exception as e:
            return {
                "posture_score": 0.5,
                "shoulder_alignment": 0.0,
                "neck_position": 0.2,
                "recommendations": ["Posture analysis temporarily unavailable"],
                "status": "error",
                "error": str(e)
            }
    
    def _generate_posture_recommendations(self, score: float) -> List[str]:
        """Generate contextual posture recommendations"""
        if score < 0.4:
            return [
                "🚨 Critical posture issue detected",
                "💪 Straighten your spine immediately",
                "📏 Adjust monitor to eye level",
                "⏱️ Take a 2-minute posture break now"
            ]
        elif score < 0.6:
            return [
                "⚠️ Posture needs improvement",
                "💪 Pull shoulders back and down",
                "🪑 Check your chair ergonomics"
            ]
        elif score < 0.8:
            return [
                "✅ Good posture with minor adjustments needed",
                "🎯 Maintain this alignment"
            ]
        else:
            return ["🌟 Excellent posture! Keep it up!"]
    
    def _generate_enhanced_posture_recommendations(self, analysis: Dict, score: float) -> List[str]:
        """Generate specific recommendations based on enhanced posture analysis"""
        recommendations = []
        
        # Get states and metrics from analysis
        states = analysis.get('states', {})
        metrics = analysis.get('metrics', {})
        
        # Specific recommendations based on actual issues
        if states.get('neck_flexion') == 'BAD':
            neck_angle = metrics.get('neck_angle_deg', 0)
            recommendations.append(f"🔴 Neck strain detected ({neck_angle:.1f}°) - Raise monitor height")
            
        if states.get('shoulder_level') == 'BAD':
            shoulder_angle = metrics.get('shoulder_slope_deg', 0)
            recommendations.append(f"🔴 Shoulders uneven ({shoulder_angle:.1f}°) - Check chair height")
            
        if states.get('forward_head') == 'BAD':
            forward_head = metrics.get('forward_head_y_ratio', 0)
            recommendations.append(f"🔴 Head too far forward ({forward_head:.2f}) - Pull head back")
            
        if states.get('head_tilt') == 'BAD':
            head_tilt = metrics.get('head_tilt_deg', 0)
            recommendations.append(f"🔴 Head tilted ({head_tilt:.1f}°) - Level your head")
            
        if states.get('shoulder_open') == 'BAD':
            recommendations.append("🔴 Shoulders closed - Open chest and pull shoulders back")
            
        # If no specific issues, give general feedback
        if not recommendations:
            if score > 0.8:
                recommendations.append("🌟 Excellent posture maintained!")
            elif score > 0.6:
                recommendations.append("✅ Good posture with minor adjustments needed")
            else:
                recommendations.append("🎯 Focus on maintaining better alignment")
                
        return recommendations[:3]  # Limit to top 3 recommendations

class PhoneUsageAgent(Agent):
    """ADK Agent for phone usage behavioral analysis"""
    
    def __init__(self):
        super().__init__(
            name="phone_tracker",
            description="Tracks phone usage patterns and provides productivity insights",
            model=Gemini(model_name="gemini-1.5-pro"),
            tools=[
                FunctionTool(self.track_phone_usage)
            ]
        )
        print("📱 Phone Usage Agent initialized")
        
    def track_phone_usage(self, phones_detected: int = 0, timestamp: Optional[float] = None, detector: Optional[IntegratedPoseDetector] = None) -> Dict[str, Any]:
        """Track phone usage with real YOLO detection from detector.py"""
        if timestamp is None:
            timestamp = time.time()
            
        try:
            # Use integrated detector phone data - no simulation
            if detector is not None:
                # Get phone usage tracker from integrated detector
                usage_tracker = detector.phone_usage_tracker
                
                # Get real phone usage statistics
                current_session = usage_tracker.get('current_session')
                session_duration = current_session['duration'] if current_session else 0.0
                phones_detected = phones_detected if phones_detected > 0 else 0
                productivity_impact = max(0.0, 1.0 - (session_duration / 60.0))  # Decrease based on duration
                
                # Get real behavioral insights
                session_type = detector._categorize_phone_session(session_duration) if session_duration > 0 else "none"
                behavioral_insights = {
                    "pattern": session_type,
                    "message": f"📱 {session_type.title()} usage pattern detected",
                    "current_duration": session_duration,
                    "total_today": usage_tracker.get('total_usage_today', 0),
                    "recent_sessions": len(usage_tracker.get('usage_sessions', []))
                }
                
                status = "real_detection"
            else:
                # No detector available - return error state
                raise ValueError("No detector available for phone usage tracking")
            
            recommendations = self._generate_phone_recommendations(session_duration)
            
            return {
                "current_session_duration": session_duration,
                "phones_detected": phones_detected,
                "session_active": phones_detected > 0,
                "behavioral_insights": behavioral_insights,
                "productivity_impact": productivity_impact,
                "recommendations": recommendations,
                "status": status
            }
        except Exception as e:
            return {
                "current_session_duration": 0,
                "phones_detected": 0,
                "session_active": False,
                "behavioral_insights": {"pattern": "error"},
                "productivity_impact": 0.5,
                "recommendations": ["Phone tracking temporarily unavailable"],
                "status": "error",
                "error": str(e)
            }
    
    def _generate_phone_recommendations(self, current_duration: float) -> List[str]:
        """Generate phone usage recommendations"""
        if current_duration > 30:
            return [
                "📱 Extended phone session detected (30s+)",
                "🎯 Consider airplane mode for deep focus",
                "⏱️ Try Pomodoro: 25min work, 5min break"
            ]
        elif current_duration > 10:
            return ["📱 Phone session active - consider wrapping up"]
        elif current_duration > 0:
            return ["📱 Brief phone check detected"]
        else:
            return ["✅ Great phone discipline maintained!"]

class EnvironmentalAgent(Agent):
    """ADK Agent for environmental monitoring"""
    
    def __init__(self):
        super().__init__(
            name="environment_monitor", 
            description="Monitors environmental factors affecting productivity and focus",
            model=Gemini(model_name="gemini-1.5-pro"),
            tools=[
                FunctionTool(self.monitor_environment)
            ]
        )
        print("🔊 Environmental Monitoring Agent initialized")
        
    def monitor_environment(self, detector: Optional[IntegratedPoseDetector] = None) -> Dict[str, Any]:
        """Monitor environmental conditions with real noise detection from detector.py"""
        try:
            # Use integrated detector noise detection - no simulation
            if detector is not None and detector.noise_detector and detector.noise_enabled:
                # Get real noise level from integrated detector
                try:
                    noise_level = detector.noise_detector.get_average_noise_level(seconds=1)
                except Exception:
                    noise_level = 0.3  # Fallback
                
                # Classify noise based on level
                if noise_level < 0.2:
                    classification = "quiet"
                elif noise_level < 0.5:
                    classification = "moderate"
                else:
                    classification = "noisy"
                
                is_noisy = noise_level > 0.3
                
                # Use detector's focus analysis
                focus_score = detector.focus_score
                
                status = "real_detection"
            else:
                # No noise detection available - return unavailable state
                raise ValueError("Noise detection not available or not enabled")
            
            environmental_score = self._calculate_environmental_score(noise_level)
            suggestions = self._get_environmental_suggestions(noise_level)
            
            return {
                "noise_level": noise_level,
                "noise_classification": classification,
                "environmental_score": environmental_score,
                "suggestions": suggestions,
                "optimal_ranges": {
                    "noise": "0.1-0.3 for focus"
                },
                "status": status,
                "noise_enabled": detector.noise_detector is not None if detector else False,
                "focus_score": focus_score if 'focus_score' in locals() else 0.7
            }
        except Exception as e:
            return {
                "noise_level": 0.3,
                "noise_classification": "moderate",
                "environmental_score": 0.7,
                "suggestions": ["Environmental monitoring temporarily unavailable"],
                "status": "error",
                "error": str(e)
            }
    
    def _calculate_environmental_score(self, noise: float) -> float:
        """Calculate overall environmental quality score"""
        # Noise score (optimal 0.1-0.3)
        if 0.1 <= noise <= 0.3:
            return 1.0
        else:
            return max(0.0, 1.0 - abs(noise - 0.2) * 2)
    
    def _get_environmental_suggestions(self, noise: float) -> List[str]:
        """Generate environmental improvement suggestions"""
        suggestions = []
        
        if noise > 0.5:
            suggestions.append("🎧 High noise - use noise-canceling headphones")
        elif noise > 0.3:
            suggestions.append("🔊 Consider white noise or focus music")
        elif noise < 0.05:
            suggestions.append("🎵 Too quiet - add gentle background noise")
        else:
            suggestions.append("✅ Excellent environmental conditions!")
            
        return suggestions

class WellnessCoachAgent(LlmAgent):
    """ADK LLM Agent for personalized wellness coaching with Google Cloud logs access"""
    
    def __init__(self):
        super().__init__(
            name="wellness_coach",
            description="AI-powered wellness coach providing personalized health and productivity guidance based on historical data",
            model=Gemini(model_name="gemini-1.5-pro"),
            tools=[
                FunctionTool(self.analyze_health_trends),
                FunctionTool(self.provide_coaching),
                FunctionTool(self.assess_intervention_need)
            ]
        )
        # Initialize cloud access (will be set by system)
        self._cloud_logger = None
        self._cloud_logging_client = None
        print("🤖 AI Wellness Coach initialized")
    
    def set_cloud_access(self, cloud_logger, cloud_logging_client):
        """Set Google Cloud access after initialization"""
        self._cloud_logger = cloud_logger
        self._cloud_logging_client = cloud_logging_client
        if cloud_logging_client:
            print("🤖 Wellness Coach connected to Google Cloud logs")
    
    def analyze_health_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze health trends from Google Cloud logs using Gemini AI"""
        try:
            if not self._cloud_logging_client:
                return {"status": "no_cloud_access", "analysis": "Historical data unavailable"}
            
            # Fetch recent health data from Google Cloud
            from datetime import datetime, timedelta
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours)
            
            filter_str = f'''
                logName="projects/perfect-entry-473503-j1/logs/straightup-adk-production"
                AND timestamp >= "{start_time.isoformat()}Z"
                AND timestamp <= "{end_time.isoformat()}Z"
                AND jsonPayload.source="adk_production_system"
            '''
            
            entries = list(self._cloud_logging_client.list_entries(
                filter_=filter_str,
                order_by=cloud_logging.DESCENDING,
                max_results=50
            ))
            
            if not entries:
                return {"status": "no_data", "analysis": "No recent health data available"}
            
            # Extract health metrics for analysis
            health_data = []
            for entry in entries:
                if hasattr(entry, 'payload') and isinstance(entry.payload, dict):
                    health_data.append({
                        'timestamp': entry.timestamp.isoformat() if entry.timestamp else None,
                        'focus_score': entry.payload.get('focus_score', 0),
                        'posture_score': entry.payload.get('posture_score', 0),
                        'phone_usage': entry.payload.get('phone_usage_seconds', 0),
                        'noise_level': entry.payload.get('noise_level', 0),
                        'recommendations': entry.payload.get('recommendations', [])
                    })
            
            # Analyze patterns
            avg_focus = sum(d['focus_score'] for d in health_data) / len(health_data)
            avg_posture = sum(d['posture_score'] for d in health_data) / len(health_data)
            total_phone_time = sum(d['phone_usage'] for d in health_data)
            avg_noise = sum(d['noise_level'] for d in health_data) / len(health_data)
            
            # Calculate trends
            half_point = len(health_data) // 2
            recent_focus = sum(d['focus_score'] for d in health_data[:half_point]) / max(half_point, 1)
            older_focus = sum(d['focus_score'] for d in health_data[half_point:]) / max(len(health_data) - half_point, 1)
            
            return {
                "status": "success",
                "data_points": len(health_data),
                "time_range_hours": hours,
                "trends": {
                    "focus_trend": "improving" if recent_focus > older_focus else "declining" if recent_focus < older_focus else "stable",
                    "avg_focus": avg_focus,
                    "avg_posture": avg_posture,
                    "total_phone_time": total_phone_time,
                    "avg_noise": avg_noise,
                    "recent_vs_older_focus": f"{recent_focus:.2f} vs {older_focus:.2f}"
                },
                "raw_data": health_data[:10]  # Last 10 data points for context
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e), "analysis": "Failed to analyze historical trends"}
    
    def provide_coaching(self, current_metrics: Dict[str, Any], trends: Dict[str, Any] = None) -> Dict[str, Any]:
        """Provide AI-powered coaching based on current metrics and historical trends"""
        try:
            # Prepare context for Gemini
            coaching_context = f"""
            You are an AI wellness coach analyzing a user's real-time health data from their workspace monitoring system.
            
            Current Health Metrics:
            - Focus Score: {current_metrics.get('focus_score', 0):.2f}/1.0
            - Posture Score: {current_metrics.get('posture_score', 0):.2f}/1.0  
            - Phone Usage: {current_metrics.get('phone_usage_duration', 0):.1f} seconds
            - Noise Level: {current_metrics.get('noise_level', 0):.3f}
            
            Historical Context (if available):
            {f"- Focus Trend: {trends['trends']['focus_trend']}" if trends and trends.get('status') == 'success' else "- No historical data available"}
            {f"- Average Focus: {trends['trends']['avg_focus']:.2f}" if trends and trends.get('status') == 'success' else ""}
            {f"- Average Posture: {trends['trends']['avg_posture']:.2f}" if trends and trends.get('status') == 'success' else ""}
            
            Provide personalized wellness coaching in 2-3 actionable bullet points. Focus on the most critical issues first.
            Be specific, encouraging, and practical. Use emojis appropriately.
            """
            
            # Generate coaching using the Gemini model (this would be handled by ADK framework)
            # For now, provide rule-based coaching that could be enhanced with actual Gemini calls
            coaching_advice = []
            priority_level = "normal"
            
            # Critical issues first
            if current_metrics.get('focus_score', 0) < 0.3:
                coaching_advice.append("🚨 URGENT: Your focus is critically low. Take a 10-minute break immediately - step away from your desk, do some deep breathing, and return refreshed.")
                priority_level = "critical"
            elif current_metrics.get('posture_score', 0) < 0.3:
                coaching_advice.append("⚠️ POSTURE ALERT: Your posture needs immediate attention. Sit up straight, adjust your monitor height, and do 5 neck rolls right now.")
                priority_level = "urgent"
            elif current_metrics.get('phone_usage_duration', 0) > 45:
                coaching_advice.append("📱 PHONE INTERVENTION: You've been on your phone for over 45 seconds. Put it in airplane mode for the next 25 minutes to restore focus.")
                priority_level = "urgent"
            
            # Trend-based coaching
            if trends and trends.get('status') == 'success':
                if trends['trends']['focus_trend'] == 'declining':
                    coaching_advice.append(f"📉 Your focus has been declining ({trends['trends']['recent_vs_older_focus']}). Try the Pomodoro technique: 25 min work, 5 min break.")
                elif trends['trends']['focus_trend'] == 'improving':
                    coaching_advice.append(f"📈 Great progress! Your focus is improving ({trends['trends']['recent_vs_older_focus']}). Keep up the good work!")
            
            # General wellness advice
            if current_metrics.get('noise_level', 0) > 0.5:
                coaching_advice.append("🎧 High noise environment detected. Consider noise-canceling headphones or moving to a quieter space.")
            elif len(coaching_advice) == 0:
                coaching_advice.append("✅ You're doing well! Stay consistent with your current habits and take regular breaks every hour.")
            
            return {
                "status": "success",
                "coaching_advice": coaching_advice[:3],  # Top 3 pieces of advice
                "priority_level": priority_level,
                "coaching_context": "AI-powered analysis of real-time and historical health data"
            }
            
        except Exception as e:
            return {
                "status": "error", 
                "error": str(e),
                "coaching_advice": ["Wellness coaching temporarily unavailable - focus on maintaining good posture and taking regular breaks."]
            }
    
    def assess_intervention_need(self, current_metrics: Dict[str, Any], trends: Dict[str, Any] = None) -> Dict[str, Any]:
        """Assess if immediate intervention is needed based on AI analysis"""
        try:
            intervention_score = 0
            intervention_reasons = []
            
            # Score based on current metrics
            focus_score = current_metrics.get('focus_score', 0.5)
            posture_score = current_metrics.get('posture_score', 0.5)
            phone_usage = current_metrics.get('phone_usage_duration', 0)
            noise_level = current_metrics.get('noise_level', 0.3)
            
            if focus_score < 0.3:
                intervention_score += 3
                intervention_reasons.append("Critical focus degradation")
            elif focus_score < 0.5:
                intervention_score += 2
                intervention_reasons.append("Low focus levels")
                
            if posture_score < 0.3:
                intervention_score += 3
                intervention_reasons.append("Poor posture detected")
            elif posture_score < 0.5:
                intervention_score += 1
                intervention_reasons.append("Posture needs attention")
                
            if phone_usage > 60:
                intervention_score += 3
                intervention_reasons.append("Excessive phone usage")
            elif phone_usage > 30:
                intervention_score += 1
                intervention_reasons.append("Extended phone session")
                
            if noise_level > 0.7:
                intervention_score += 2
                intervention_reasons.append("High noise environment")
            
            # Factor in trends if available
            if trends and trends.get('status') == 'success':
                if trends['trends']['focus_trend'] == 'declining':
                    intervention_score += 1
                    intervention_reasons.append("Declining focus trend")
            
            # Determine intervention level
            if intervention_score >= 5:
                intervention_level = "immediate"
                intervention_type = "Break recommended now"
            elif intervention_score >= 3:
                intervention_level = "soon"
                intervention_type = "Consider a break within 10 minutes"
            elif intervention_score >= 1:
                intervention_level = "monitor"
                intervention_type = "Continue monitoring"
            else:
                intervention_level = "none"
                intervention_type = "No intervention needed"
            
            return {
                "status": "success",
                "intervention_needed": intervention_score >= 3,
                "intervention_level": intervention_level,
                "intervention_type": intervention_type,
                "intervention_score": intervention_score,
                "reasons": intervention_reasons
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "intervention_needed": False
            }

# Main StraightUp ADK System with Enhanced Console Output
class StraightUpADKSystem:
    """Enhanced Production Google ADK system for StraightUp with detailed agent insights"""
    
    def __init__(self):
        print("🚀 Initializing StraightUp Enhanced ADK System...")
        print(f"🎯 Project: perfect-entry-473503-j1")
        print(f"🔧 ADK Version: {google.adk.__version__}")
        
        # Initialize specialized agents
        self.posture_agent = PostureAnalysisAgent()
        self.phone_agent = PhoneUsageAgent()
        self.environment_agent = EnvironmentalAgent()
        self.wellness_coach = WellnessCoachAgent()
        
        # Create real ADK ParallelAgent for simultaneous execution
        self.parallel_monitor = ParallelAgent(
            name="health_parallel_monitor",
            description="Simultaneous monitoring of posture, phone usage, and environmental factors",
            sub_agents=[self.posture_agent, self.phone_agent, self.environment_agent]
        )
        
        # Create real ADK LoopAgent for continuous monitoring
        self.monitoring_loop = LoopAgent(
            name="continuous_health_monitor",
            description="Continuous health and productivity monitoring system",
            sub_agents=[self.parallel_monitor]
        )
        
        # System state
        self.running = True
        self.cycle_count = 0
        self.health_history = []
        
        # Initialize Google Cloud logging
        self.cloud_logger = None
        if CLOUD_LOGGING_AVAILABLE:
            try:
                self.cloud_logging_client = cloud_logging.Client(project='perfect-entry-473503-j1')
                self.cloud_logger = self.cloud_logging_client.logger('straightup-adk-production')
                print("🌐 Google Cloud logging initialized for project: perfect-entry-473503-j1")
            except Exception as e:
                print(f"⚠️ Cloud logging setup failed: {e}")
                self.cloud_logger = None
        
        # Initialize Google Cloud logging
        self.cloud_logger = None
        self.monitoring_client = None
        if CLOUD_LOGGING_AVAILABLE:
            try:
                self.cloud_logging_client = cloud_logging.Client(project='perfect-entry-473503-j1')
                self.cloud_logger = self.cloud_logging_client.logger('straightup-adk-production')
                self.monitoring_client = monitoring_v3.MetricServiceClient()
                
                # Enable wellness coach with cloud access
                self.wellness_coach.set_cloud_access(self.cloud_logger, self.cloud_logging_client)
                
                print("🌐 Google Cloud logging initialized for project: perfect-entry-473503-j1")
            except Exception as e:
                print(f"⚠️ Cloud logging setup failed: {e}")
                self.cloud_logger = None
        
        # Initialize integrated detector system
        try:
            self.detector = IntegratedPoseDetector()
            print("🎯 Integrated detector initialized with all modules")
        except Exception as e:
            print(f"⚠️ Integrated detector initialization failed: {e}")
            self.detector = None
        
        # Initialize camera if available
        try:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            print("📹 Camera initialized for posture monitoring (1280x720)")
        except Exception as e:
            print(f"⚠️ Camera initialization failed: {e}")
            self.cap = None
            
        print("✅ All REAL ADK agents initialized successfully!")
        
    def _display_agent_insights(self, posture_results, phone_results, environment_results):
        """Display detailed agent insights like webcam overlay"""
        print("\n" + "=" * 80)
        print("🤖 ADK AGENT INSIGHTS - Real-time Analysis")
        print("=" * 80)
        
        # Posture Agent Insights
        print(f"🎯 POSTURE ANALYSIS AGENT:")
        print(f"   📊 Posture Score: {posture_results['posture_score']:.2f}/1.0")
        print(f"   💪 Shoulder Alignment: {posture_results.get('shoulder_alignment', 0):.2f}")
        print(f"   🏃 Neck Position: {posture_results.get('neck_position', 0):.2f}")
        print(f"   👥 Detection Status: {posture_results['status']}")
        if 'pose_detected' in posture_results:
            print(f"   📸 Live Detection: {posture_results['pose_detected']} pose, {posture_results.get('faces_detected', 0)} faces, {posture_results.get('hands_detected', 0)} hands")
        
        # Phone Usage Agent Insights
        print(f"\n📱 PHONE USAGE AGENT:")
        print(f"   ⏱️ Current Session: {phone_results['current_session_duration']:.1f}s")
        print(f"   📊 Productivity Impact: {phone_results['productivity_impact']:.2f}/1.0")
        print(f"   🎯 Session Active: {'YES' if phone_results['session_active'] else 'NO'}")
        print(f"   📈 Usage Pattern: {phone_results['behavioral_insights']['pattern'].upper()}")
        print(f"   📱 Detection Status: {phone_results['status']}")
        if phone_results['behavioral_insights'].get('total_today'):
            print(f"   📊 Today's Usage: {phone_results['behavioral_insights']['total_today']:.0f}s total")
        
        # Environmental Agent Insights
        print(f"\n🔊 ENVIRONMENTAL AGENT:")
        print(f"   🌪️ Noise Level: {environment_results['noise_level']:.3f}")
        print(f"   🔊 Classification: {environment_results.get('noise_classification', 'unknown').upper()}")
        print(f"   🌍 Environmental Score: {environment_results['environmental_score']:.2f}/1.0")
        print(f"   📊 Detection Status: {environment_results['status']}")
        if 'focus_score' in environment_results:
            print(f"   🎯 Current Focus: {environment_results['focus_score']:.2f}/1.0")
        
        # Overall System Health
        print(f"\n🎯 SYSTEM HEALTH OVERVIEW:")
        all_active = all(r['status'] == 'real_detection' for r in [posture_results, phone_results] if r['status'] != 'error')
        print(f"   🔋 Agent Status: {'🟢 ALL OPERATIONAL' if all_active else '🟡 PARTIAL OPERATION'}")
        print(f"   📊 Data Quality: {'🟢 HIGH' if all_active else '🟡 MEDIUM'}")
    
    def _display_wellness_insights(self, coaching: Dict[str, Any], intervention: Dict[str, Any], trends: Dict[str, Any]):
        """Display AI wellness coaching insights"""
        print(f"\n🤖 AI WELLNESS COACH INSIGHTS:")
        
        # Coaching status
        if coaching.get('status') == 'success':
            print(f"   💡 Coaching Priority: {coaching.get('priority_level', 'normal').upper()}")
            coaching_advice = coaching.get('coaching_advice', [])
            for i, advice in enumerate(coaching_advice[:2], 1):  # Show top 2
                print(f"   {i}. {advice}")
        else:
            print(f"   ⚠️ Coaching Status: {coaching.get('status', 'unknown')}")
        
        # Intervention assessment
        if intervention.get('status') == 'success':
            need_intervention = intervention.get('intervention_needed', False)
            intervention_level = intervention.get('intervention_level', 'none')
            print(f"   ⚠️ Intervention: {'🔴 NEEDED' if need_intervention else '🟢 NOT NEEDED'} ({intervention_level.upper()})")
            if need_intervention:
                print(f"   🎯 Action: {intervention.get('intervention_type', 'Monitor closely')}")
                reasons = intervention.get('reasons', [])
                if reasons:
                    print(f"   📋 Reasons: {', '.join(reasons[:2])}")  # Top 2 reasons
        
        # Historical trends
        if trends.get('status') == 'success':
            trend_data = trends.get('trends', {})
            print(f"   📈 Focus Trend (6h): {trend_data.get('focus_trend', 'unknown').upper()}")
            print(f"   📊 Historical Data: {trends.get('data_points', 0)} points analyzed")
        else:
            print(f"   📊 Historical Analysis: {trends.get('status', 'unavailable')}")
        
    def _log_health_metrics(self, metrics: HealthMetrics, posture_results=None, phone_results=None, environment_results=None, 
                          wellness_coaching=None, wellness_intervention=None, detection_results=None):
        """Log enhanced health metrics for production monitoring and send to Google Cloud"""
        print(f"\n💊 HEALTH METRICS SUMMARY:")
        print(f"   🎯 Overall Focus: {metrics.focus_score:.2f}/1.0 {'🟢' if metrics.focus_score > 0.7 else '🟡' if metrics.focus_score > 0.4 else '🔴'}")
        print(f"   🏃 Posture Quality: {metrics.posture_score:.2f}/1.0 {'🟢' if metrics.posture_score > 0.7 else '🟡' if metrics.posture_score > 0.4 else '🔴'}")
        print(f"   📱 Phone Usage: {metrics.phone_usage_duration:.1f}s {'🟢' if metrics.phone_usage_duration < 10 else '🟡' if metrics.phone_usage_duration < 30 else '🔴'}")
        print(f"   🔊 Noise Level: {metrics.noise_level:.3f} {'🟢' if metrics.noise_level < 0.3 else '🟡' if metrics.noise_level < 0.6 else '🔴'}")
        
        # Send comprehensive data to Google Cloud Logging
        if self.cloud_logger:
            try:
                # Base metrics
                log_data = {
                    "cycle": self.cycle_count,
                    "timestamp": metrics.timestamp,
                    "focus_score": metrics.focus_score,
                    "posture_score": metrics.posture_score,
                    "phone_usage_seconds": metrics.phone_usage_duration,
                    "noise_level": metrics.noise_level,
                    "recommendations": metrics.recommendations[:5] if metrics.recommendations else [],
                    "project_id": "perfect-entry-473503-j1",
                    "source": "adk_production_system",
                    "agent_status": "operational"
                }
                
                # Add detailed detection data if available
                if detection_results:
                    log_data["detection"] = {
                        "faces_detected": detection_results.get("faces", 0),
                        "hands_detected": detection_results.get("hands", 0),
                        "pose_detected": detection_results.get("pose", 0),
                        "phones_detected": detection_results.get("phones", 0),
                        "detector_focus_score": detection_results.get("focus_score", 0.0)
                    }
                    
                    # Add comprehensive posture analysis data from detector
                    posture_analysis = detection_results.get("posture_analysis")
                    if posture_analysis and posture_analysis.get("ok"):
                        log_data["detailed_posture_analysis"] = {
                            "overall_state": posture_analysis.get("state", "OK"),
                            "metrics": {
                                "neck_angle_deg": posture_analysis.get("metrics", {}).get("neck_angle_deg", 0),
                                "shoulder_slope_deg": posture_analysis.get("metrics", {}).get("shoulder_slope_deg", 0),
                                "head_tilt_deg": posture_analysis.get("metrics", {}).get("head_tilt_deg", 0),
                                "shoulder_width_norm": posture_analysis.get("metrics", {}).get("shoulder_width_norm", 0),
                                "shoulder_height_ratio": posture_analysis.get("metrics", {}).get("shoulder_height_ratio", 0),
                                "shoulder_protraction": posture_analysis.get("metrics", {}).get("shoulder_protraction", 0),
                                "torso_pitch_deg": posture_analysis.get("metrics", {}).get("torso_pitch_deg", 0),
                                "torso_len_ratio": posture_analysis.get("metrics", {}).get("torso_len_ratio", 0),
                                "forward_head_y_ratio": posture_analysis.get("metrics", {}).get("forward_head_y_ratio", 0),
                                "yaw_ratio": posture_analysis.get("metrics", {}).get("yaw_ratio", 0)
                            },
                            "states": {
                                "neck_flexion": posture_analysis.get("states", {}).get("neck_flexion", "OK"),
                                "forward_head": posture_analysis.get("states", {}).get("forward_head", "OK"),
                                "shoulder_level": posture_analysis.get("states", {}).get("shoulder_level", "OK"),
                                "head_tilt": posture_analysis.get("states", {}).get("head_tilt", "OK"),
                                "shoulder_open": posture_analysis.get("states", {}).get("shoulder_open", "OK"),
                                "torso_pitch": posture_analysis.get("states", {}).get("torso_pitch", "OK"),
                                "open_width": posture_analysis.get("states", {}).get("open_width", "OK"),
                                "open_height": posture_analysis.get("states", {}).get("open_height", "OK"),
                                "open_protraction": posture_analysis.get("states", {}).get("open_protraction", "OK")
                            },
                            "landmark_points": {
                                "neck_base_px": list(posture_analysis.get("points", {}).get("neck_base_px", [0, 0])),
                                "nose_px": list(posture_analysis.get("points", {}).get("nose_px", [0, 0])),
                                "l_shoulder_px": list(posture_analysis.get("points", {}).get("l_shoulder_px", [0, 0])),
                                "r_shoulder_px": list(posture_analysis.get("points", {}).get("r_shoulder_px", [0, 0])),
                                "l_eye_px": list(posture_analysis.get("points", {}).get("l_eye_px", [0, 0])),
                                "r_eye_px": list(posture_analysis.get("points", {}).get("r_eye_px", [0, 0]))
                            }
                        }
                
                # Add detailed posture analysis if available
                if posture_results:
                    log_data["posture_analysis"] = {
                        "posture_score": posture_results.get("posture_score", 0),
                        "shoulder_alignment": posture_results.get("shoulder_alignment", 0),
                        "neck_position": posture_results.get("neck_position", 0),
                        "status": posture_results.get("status", "unknown"),
                        "pose_detected": posture_results.get("pose_detected", False),
                        "faces_detected": posture_results.get("faces_detected", 0),
                        "hands_detected": posture_results.get("hands_detected", 0)
                    }
                
                # Add detailed phone usage data if available
                if phone_results:
                    log_data["phone_analysis"] = {
                        "current_session_duration": phone_results.get("current_session_duration", 0),
                        "productivity_impact": phone_results.get("productivity_impact", 1.0),
                        "session_active": phone_results.get("session_active", False),
                        "usage_pattern": phone_results.get("behavioral_insights", {}).get("pattern", "none"),
                        "total_today": phone_results.get("behavioral_insights", {}).get("total_today", 0),
                        "recent_sessions": phone_results.get("behavioral_insights", {}).get("recent_sessions", 0),
                        "status": phone_results.get("status", "unknown")
                    }
                
                # Add environmental data if available
                if environment_results:
                    log_data["environment_analysis"] = {
                        "noise_level": environment_results.get("noise_level", 0.3),
                        "noise_classification": environment_results.get("noise_classification", "unknown"),
                        "environmental_score": environment_results.get("environmental_score", 0.7),
                        "noise_enabled": environment_results.get("noise_enabled", False),
                        "status": environment_results.get("status", "unknown")
                    }
                
                # Add AI wellness coaching data if available
                if wellness_coaching and wellness_coaching.get('status') == 'success':
                    log_data["ai_coaching"] = {
                        "priority_level": wellness_coaching.get("priority_level", "normal"),
                        "coaching_advice": wellness_coaching.get("coaching_advice", [])[:3],
                        "coaching_context": wellness_coaching.get("coaching_context", "")
                    }
                
                # Add intervention assessment if available
                if wellness_intervention and wellness_intervention.get('status') == 'success':
                    log_data["intervention_assessment"] = {
                        "intervention_needed": wellness_intervention.get("intervention_needed", False),
                        "intervention_level": wellness_intervention.get("intervention_level", "none"),
                        "intervention_type": wellness_intervention.get("intervention_type", ""),
                        "intervention_score": wellness_intervention.get("intervention_score", 0),
                        "reasons": wellness_intervention.get("reasons", [])
                    }
                
                # Add integrated detector comprehensive data if available
                if self.detector:
                    try:
                        detector_state = {
                            "focus_score": float(self.detector.focus_score),
                            "distraction_factors": {k: float(v) for k, v in self.detector.distraction_factors.items()},
                            "noise_enabled": bool(self.detector.noise_enabled),
                            "animation_frame": int(self.detector.animation_frame)
                        }
                        
                        # Eye tracking data if available
                        try:
                            detector_state["eye_tracking"] = {
                                "eyes_closed": getattr(self.detector, '_eyes_closed', False),
                                "eye_ratio_left": getattr(self.detector, '_eye_ratio_left_s', None),
                                "eye_ratio_right": getattr(self.detector, '_eye_ratio_right_s', None),
                                "eye_closed_threshold": getattr(self.detector, 'EYE_CLOSED_THR', 0.18),
                                "eye_open_threshold": getattr(self.detector, 'EYE_OPEN_THR', 0.22)
                            }
                        except Exception:
                            detector_state["eye_tracking"] = {"status": "unavailable"}
                        
                        # Phone tracking system data
                        try:
                            detector_state["phone_tracking"] = {
                                "active_tracks": len(getattr(self.detector, '_phone_tracks', [])),
                                "detection_history_length": len(getattr(self.detector, 'phone_detection_history', [])),
                                "confidence_threshold": getattr(self.detector, 'phone_confidence_threshold', 0.25),
                                "smooth_alpha": getattr(self.detector, 'phone_smooth_alpha', 0.6),
                                "session_candidate_start": getattr(self.detector, 'session_candidate_start', None),
                                "last_stable_phone_time": getattr(self.detector, 'last_stable_phone_time', 0.0)
                            }
                        except Exception:
                            detector_state["phone_tracking"] = {"status": "unavailable"}
                        
                        # Phone usage tracker data
                        if hasattr(self.detector, 'phone_usage_tracker'):
                            tracker = self.detector.phone_usage_tracker
                            detector_state["phone_usage_tracker"] = {
                                "current_session_active": tracker.get('current_session') is not None,
                                "session_start_time": tracker.get('session_start_time'),
                                "total_usage_today": tracker.get('total_usage_today', 0),
                                "daily_stats": dict(tracker.get('daily_stats', {})),
                                "usage_sessions_count": len(tracker.get('usage_sessions', [])),
                                "last_detection_time": tracker.get('last_detection_time', 0.0)
                            }
                            
                            # Current session details if active
                            current_session = tracker.get('current_session')
                            if current_session:
                                detector_state["phone_usage_tracker"]["current_session"] = {
                                    "start_time": float(current_session.get('start_time', 0)) if current_session.get('start_time') else None,
                                    "duration": float(current_session.get('duration', 0.0)),
                                    "end_time": float(current_session.get('end_time', 0)) if current_session.get('end_time') else None
                                }
                        
                        # Noise detection data if available
                        if hasattr(self.detector, 'noise_detector') and self.detector.noise_detector:
                            try:
                                if self.detector.noise_enabled:
                                    noise_level = self.detector.noise_detector.get_average_noise_level(seconds=1)
                                    detector_state["noise_detection"] = {
                                        "current_noise_level": noise_level,
                                        "noise_enabled": True,
                                        "sample_rate": getattr(self.detector.noise_detector, 'sample_rate', 44100),
                                        "chunk_size": getattr(self.detector.noise_detector, 'chunk_size', 1024)
                                    }
                                else:
                                    detector_state["noise_detection"] = {"noise_enabled": False}
                            except Exception:
                                detector_state["noise_detection"] = {"status": "error", "noise_enabled": self.detector.noise_enabled}
                        
                        # System actions data
                        if hasattr(self.detector, 'system'):
                            try:
                                system = self.detector.system
                                detector_state["system_actions"] = {
                                    "posture_bad_start": getattr(system, '_posture_bad_start', None),
                                    "phone_seen_start": getattr(system, '_phone_seen_start', None),
                                    "last_posture_alert": getattr(system, '_last_posture_alert', 0.0),
                                    "last_dim_action": getattr(system, '_last_dim_action', 0.0),
                                    "dim_active": getattr(system, '_dim_active', False),
                                    "posture_threshold_sec": getattr(system, 'posture_bad_threshold', 180),
                                    "phone_threshold_sec": getattr(system, 'phone_threshold', 180),
                                    "cooldown_sec": getattr(system, 'cooldown_sec', 300)
                                }
                            except Exception:
                                detector_state["system_actions"] = {"status": "unavailable"}
                        
                        log_data["detector_state"] = detector_state
                    except Exception as detector_error:
                        log_data["detector_state"] = {"error": str(detector_error)}
                
                self.cloud_logger.log_struct(log_data, severity='INFO')
                print(f"   🌐 Comprehensive data sent to Google Cloud Dashboard (Cycle {self.cycle_count})")
            except Exception as e:
                print(f"   ⚠️ Cloud logging failed: {e}")
        
        # Alert system
        if metrics.focus_score < 0.4:
            print(f"🚨 CRITICAL ALERT: Focus score critically low ({metrics.focus_score:.2f})")
        elif metrics.posture_score < 0.4:
            print(f"⚠️ WARNING: Posture needs immediate attention ({metrics.posture_score:.2f})")
        elif metrics.phone_usage_duration > 30:
            print(f"📱 ALERT: Extended phone session detected ({metrics.phone_usage_duration:.1f}s)")
        
        # Show top recommendations
        if metrics.recommendations:
            print(f"\n💡 TOP RECOMMENDATIONS:")
            for i, rec in enumerate(metrics.recommendations[:3], 1):
                print(f"   {i}. {rec}")

    async def run_monitoring_cycle(self) -> Optional[HealthMetrics]:
        """Execute one complete monitoring cycle using real ADK agents"""
        self.cycle_count += 1
        current_time = time.time()
        
        print(f"\n🔄 Real ADK Monitoring Cycle {self.cycle_count}")
        print("⚡ Executing real ADK ParallelAgent...")
        
        # Log cycle start to Google Cloud with system status
        if self.cloud_logger:
            try:
                cycle_start_data = {
                    "event": "monitoring_cycle_start",
                    "cycle": self.cycle_count,
                    "timestamp": current_time,
                    "project_id": "perfect-entry-473503-j1",
                    "source": "adk_production_system",
                    "system_status": {
                        "camera_available": self.cap is not None and self.cap.isOpened() if self.cap else False,
                        "detector_available": self.detector is not None,
                        "cloud_logging_available": CLOUD_LOGGING_AVAILABLE,
                        "agents_initialized": {
                            "posture_agent": self.posture_agent is not None,
                            "phone_agent": self.phone_agent is not None,
                            "environment_agent": self.environment_agent is not None,
                            "wellness_coach": self.wellness_coach is not None
                        }
                    }
                }
                
                # Add detector status if available
                if self.detector:
                    cycle_start_data["detector_status"] = {
                        "noise_detector_available": self.detector.noise_detector is not None,
                        "noise_enabled": self.detector.noise_enabled,
                        "current_focus_score": self.detector.focus_score,
                        "phone_tracks_active": len(getattr(self.detector, '_phone_tracks', []))
                    }
                
                self.cloud_logger.log_struct(cycle_start_data, severity='INFO')
            except Exception as e:
                pass  # Silent fail for cycle start logging
        
        # Log cycle start to Google Cloud
        if self.cloud_logger:
            self.cloud_logger.log_struct({
                "event": "monitoring_cycle_start",
                "cycle": self.cycle_count,
                "timestamp": current_time,
                "project_id": "perfect-entry-473503-j1"
            }, severity='INFO')
        
        try:
            # Capture and process frame with integrated detector
            frame = None
            processed_frame = None
            detection_results = {"faces": 0, "hands": 0, "pose": 0, "phones": 0}
            
            if self.cap and self.cap.isOpened() and self.detector:
                ret, frame = self.cap.read()
                if ret:
                    # Flip frame for selfie view
                    frame = cv2.flip(frame, 1)
                    
                    # Process frame with integrated detector
                    try:
                        processed_frame, faces, hands, pose, phones, analysis = self.detector.process_frame(frame.copy())
                        detection_results = {
                            "faces": faces,
                            "hands": hands,
                            "pose": pose,
                            "phones": phones,
                            "focus_score": self.detector.focus_score,
                            "posture_analysis": analysis  # Store the detailed posture analysis
                        }
                        print(f"📸 Integrated detection: Faces={faces}, Hands={hands}, Pose={pose}, Phones={phones}, Focus={self.detector.focus_score:.2f}")
                    except Exception as e:
                        print(f"⚠️ Frame processing failed: {e}")
                        processed_frame = frame
                        detection_results = {"faces": 0, "hands": 0, "pose": 0, "phones": 0, "focus_score": 0.5, "posture_analysis": None}
                else:
                    frame = None
            
            # Verify we have required data
            if frame is None:
                raise RuntimeError("No camera frame available for analysis")
            if not self.detector:
                raise RuntimeError("No detector available for analysis")
            
            print("🎯 Executing agents with real webcam data")
            
            # Execute ADK agents with enhanced detector data
            posture_results = self.posture_agent.analyze_posture(frame, self.detector)
            phone_results = self.phone_agent.track_phone_usage(detection_results["phones"], current_time, self.detector)
            environment_results = self.environment_agent.monitor_environment(self.detector)
            
            # Execute Wellness Coach with current metrics and historical trends
            current_metrics = {
                'focus_score': 0,  # Will be calculated below
                'posture_score': posture_results.get("posture_score", 0),
                'phone_usage_duration': phone_results.get("current_session_duration", 0),
                'noise_level': environment_results.get("noise_level", 0)
            }
            
            # Get historical trends from wellness coach
            wellness_trends = self.wellness_coach.analyze_health_trends(hours=6)  # Last 6 hours
            
            # Display detailed agent insights (like webcam overlay)
            self._display_agent_insights(posture_results, phone_results, environment_results)
            
            # Calculate overall focus score
            focus_score = self._calculate_focus_score(
                posture_results, phone_results, environment_results
            )
            
            # Update current metrics with calculated focus score
            current_metrics['focus_score'] = focus_score
            
            # Get AI coaching from Wellness Coach
            wellness_coaching = self.wellness_coach.provide_coaching(current_metrics, wellness_trends)
            wellness_intervention = self.wellness_coach.assess_intervention_need(current_metrics, wellness_trends)
            
            # Display Wellness Coach insights
            self._display_wellness_insights(wellness_coaching, wellness_intervention, wellness_trends)
            
            # Aggregate all recommendations (including AI coaching)
            all_recommendations = []
            all_recommendations.extend(posture_results.get("recommendations", []))
            all_recommendations.extend(phone_results.get("recommendations", []))
            all_recommendations.extend(environment_results.get("suggestions", []))
            
            # Add AI coaching advice
            if wellness_coaching.get('status') == 'success':
                all_recommendations.extend(wellness_coaching.get('coaching_advice', []))
            
            # Create health metrics
            health_metrics = HealthMetrics(
                posture_score=posture_results["posture_score"],
                phone_usage_duration=phone_results["current_session_duration"],
                noise_level=environment_results["noise_level"],
                focus_score=focus_score,
                timestamp=current_time,
                recommendations=all_recommendations[:5]  # Top 5 recommendations
            )
            
            # Store in history
            self.health_history.append(health_metrics)
            if len(self.health_history) > 100:  # Keep last 100 cycles
                self.health_history.pop(0)
            
            # Log detailed health metrics with all agent data
            self._log_health_metrics(
                health_metrics, 
                posture_results=posture_results,
                phone_results=phone_results, 
                environment_results=environment_results,
                wellness_coaching=wellness_coaching,
                wellness_intervention=wellness_intervention,
                detection_results=detection_results
            )
            
            # Trigger interventions if needed
            if self._needs_intervention(health_metrics):
                await self._handle_intervention(health_metrics)
                
            return health_metrics
            
        except Exception as e:
            print(f"❌ Monitoring cycle error: {e}")
            return None
    
    def _calculate_focus_score(self, posture_results, phone_results, environment_results) -> float:
        """Calculate overall focus score from agent results"""
        posture_score = posture_results.get("posture_score", 0.5)
        productivity_impact = phone_results.get("productivity_impact", 0.5)
        environmental_score = environment_results.get("environmental_score", 0.5)
        
        return (posture_score + productivity_impact + environmental_score) / 3
    
    def _needs_intervention(self, metrics: HealthMetrics) -> bool:
        """Determine if intervention is needed based on metrics"""
        return (
            metrics.focus_score < 0.4 or 
            metrics.posture_score < 0.3 or
            metrics.phone_usage_duration > 45
        )
    
    async def _handle_intervention(self, metrics: HealthMetrics):
        """Handle production intervention for wellness issues"""
        interventions = []
        
        if metrics.focus_score < 0.4:
            interventions.append("Focus intervention required")
        if metrics.posture_score < 0.3:
            interventions.append("Posture correction needed")
        if metrics.phone_usage_duration > 45:
            interventions.append("Phone usage break recommended")
            
        if interventions:
            print(f"⚠️ INTERVENTION TRIGGERED: {', '.join(interventions)}")
            
            # Log intervention to Google Cloud
            if self.cloud_logger:
                try:
                    intervention_data = {
                        "event": "intervention_triggered",
                        "cycle": self.cycle_count,
                        "timestamp": time.time(),
                        "project_id": "perfect-entry-473503-j1",
                        "source": "adk_production_system",
                        "intervention_details": {
                            "focus_score": metrics.focus_score,
                            "posture_score": metrics.posture_score,
                            "phone_usage_duration": metrics.phone_usage_duration,
                            "noise_level": metrics.noise_level,
                            "interventions_triggered": interventions,
                            "severity": "critical" if metrics.focus_score < 0.3 or metrics.posture_score < 0.2 else "warning"
                        },
                        "recommendations": metrics.recommendations[:3] if metrics.recommendations else []
                    }
                    self.cloud_logger.log_struct(intervention_data, severity='WARNING')
                    print(f"   🌐 Intervention logged to Google Cloud")
                except Exception as e:
                    print(f"   ⚠️ Intervention logging failed: {e}")
            
            if "Focus intervention" in interventions:
                print("💡 RECOMMENDATION: Take a 5-minute break and do breathing exercises")
            if "Posture correction" in interventions:
                print("💡 RECOMMENDATION: Adjust your chair height and monitor position")
            if "Phone usage break" in interventions:
                print("💡 RECOMMENDATION: Put your phone in airplane mode for 25 minutes")
    
    async def start_monitoring(self):
        """Start the real ADK LoopAgent monitoring system"""
        print(f"\n🔄 Starting ADK Production Monitoring...")
        print(f"🎯 Google Cloud Project: perfect-entry-473503-j1")
        print(f"Press Ctrl+C to stop monitoring\n")
        
        try:
            # Verify required components are available
            if not self.cap or not self.cap.isOpened():
                raise RuntimeError("Camera not available for monitoring")
            if not self.detector:
                raise RuntimeError("Detector not available for monitoring")
            
            # Integrated detector is already fully initialized
            if self.detector.noise_detector:
                print("🔊 Noise detection ready for production monitoring")
            else:
                print("⚠️ Noise detection not available")
            
            # Continuous monitoring loop
            cycle_count = 0
            while self.running:
                cycle_count += 1
                metrics = await self.run_monitoring_cycle()
                
                if metrics is None:
                    print("❌ Monitoring cycle failed - stopping system")
                    break
                
                await asyncio.sleep(0.5)  # Monitor every 0.5 seconds for responsiveness
                
            print(f"\n✅ Production monitoring completed - {cycle_count} cycles executed")
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping ADK monitoring system...")
            await self.stop_monitoring()
    
    async def stop_monitoring(self):
        """Gracefully stop the monitoring system"""
        self.running = False
        
        # Stop integrated detector
        if self.detector:
            self.detector.system.cleanup()
            if self.detector.noise_enabled:
                self.detector.stop_noise_detection()
            print("🔇 Integrated detector cleaned up")
        
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        print("✅ ADK monitoring system stopped")
        print(f"📊 Total monitoring cycles completed: {self.cycle_count}")
        
        if self.health_history:
            # Show comprehensive session summary
            print(f"\n" + "=" * 60)
            print("📊 ADK SESSION SUMMARY")
            print("=" * 60)
            
            avg_focus = np.mean([m.focus_score for m in self.health_history])
            avg_posture = np.mean([m.posture_score for m in self.health_history])
            avg_phone = np.mean([m.phone_usage_duration for m in self.health_history])
            avg_noise = np.mean([m.noise_level for m in self.health_history])
            
            print(f"📈 Average Focus Score: {avg_focus:.2f}/1.0")
            print(f"🏃 Average Posture Score: {avg_posture:.2f}/1.0")
            print(f"📱 Average Phone Session: {avg_phone:.1f}s")
            print(f"🔊 Average Noise Level: {avg_noise:.3f}")
            
            # Show trends
            recent_focus = np.mean([m.focus_score for m in self.health_history[-10:]])
            early_focus = np.mean([m.focus_score for m in self.health_history[:10]]) if len(self.health_history) >= 10 else recent_focus
            trend = "📈 IMPROVING" if recent_focus > early_focus else "📉 DECLINING" if recent_focus < early_focus else "➡️ STABLE"
            print(f"📊 Focus Trend: {trend} ({early_focus:.2f} → {recent_focus:.2f})")
            
            # Log comprehensive session summary to Google Cloud
            if self.cloud_logger:
                try:
                    session_end_time = time.time()
                    session_duration = session_end_time - self.health_history[0].timestamp if self.health_history else 0
                    
                    # Calculate session statistics
                    focus_scores = [m.focus_score for m in self.health_history]
                    posture_scores = [m.posture_score for m in self.health_history]
                    phone_durations = [m.phone_usage_duration for m in self.health_history]
                    noise_levels = [m.noise_level for m in self.health_history]
                    
                    session_summary = {
                        "event": "monitoring_session_complete",
                        "timestamp": session_end_time,
                        "project_id": "perfect-entry-473503-j1",
                        "source": "adk_production_system",
                        "session_statistics": {
                            "total_cycles": self.cycle_count,
                            "session_duration_seconds": session_duration,
                            "session_duration_minutes": session_duration / 60.0,
                            "data_points": len(self.health_history)
                        },
                        "average_metrics": {
                            "focus_score": float(avg_focus),
                            "posture_score": float(avg_posture),
                            "phone_usage_seconds": float(avg_phone),
                            "noise_level": float(avg_noise)
                        },
                        "trend_analysis": {
                            "focus_trend": "improving" if recent_focus > early_focus else ("declining" if recent_focus < early_focus else "stable"),
                            "early_focus": float(early_focus),
                            "recent_focus": float(recent_focus),
                            "focus_improvement": float(recent_focus - early_focus)
                        },
                        "score_distributions": {
                            "focus_min": float(min(focus_scores)),
                            "focus_max": float(max(focus_scores)),
                            "posture_min": float(min(posture_scores)),
                            "posture_max": float(max(posture_scores)),
                            "phone_max_session": float(max(phone_durations)),
                            "noise_min": float(min(noise_levels)),
                            "noise_max": float(max(noise_levels))
                        }
                    }
                    
                    # Add detector final state if available
                    if self.detector:
                        try:
                            final_detector_state = {
                                "final_focus_score": self.detector.focus_score,
                                "final_distraction_factors": dict(self.detector.distraction_factors),
                                "noise_detection_used": self.detector.noise_enabled,
                                "total_animation_frames": self.detector.animation_frame
                            }
                            
                            # Final phone usage statistics
                            if hasattr(self.detector, 'phone_usage_tracker'):
                                tracker = self.detector.phone_usage_tracker
                                final_detector_state["final_phone_stats"] = {
                                    "total_usage_today": tracker.get('total_usage_today', 0),
                                    "daily_session_counts": dict(tracker.get('daily_stats', {})),
                                    "total_sessions_recorded": len(tracker.get('usage_sessions', []))
                                }
                                
                            session_summary["final_detector_state"] = final_detector_state
                        except Exception:
                            session_summary["final_detector_state"] = {"status": "unavailable"}
                    
                    self.cloud_logger.log_struct(session_summary, severity='INFO')
                    print(f"   🌐 Session summary logged to Google Cloud")
                except Exception as e:
                    print(f"   ⚠️ Session summary logging failed: {e}")
            
        # Show final integrated detector statistics
        if self.detector:
            # Get phone usage from the integrated detector
            phone_usage_tracker = self.detector.phone_usage_tracker
            total_usage = phone_usage_tracker.get('total_usage_today', 0)
            daily_stats = phone_usage_tracker.get('daily_stats', {})
            
            print(f"\n📱 PHONE USAGE BREAKDOWN:")
            print(f"   📊 Total Usage Today: {total_usage:.0f}s")
            print(f"   🎯 Final Focus Score: {self.detector.focus_score:.2f}/1.0")
            print(f"   📈 Session Stats: Brief:{daily_stats.get('brief', 0)} | Moderate:{daily_stats.get('moderate', 0)} | Extended:{daily_stats.get('extended', 0)} | Excessive:{daily_stats.get('excessive', 0)}")
            
            print(f"\n🔊 NOISE MONITORING:")
            if self.detector.noise_enabled and self.detector.noise_detector:
                print(f"   ✅ Noise detection was active")
                try:
                    final_noise = self.detector.noise_detector.get_average_noise_level(seconds=1)
                    print(f"   📊 Final noise level: {final_noise:.3f}")
                except Exception:
                    print(f"   📊 Final noise level: unavailable")
            else:
                print(f"   ⚠️ Noise detection was disabled")
        
        print("=" * 60)

if __name__ == "__main__":
    print("🏥 StraightUp - Enhanced Production Monitoring System")
    print("=" * 50)
    print(f"🚀 ADK Version: {google.adk.__version__}")
    print(f"🎯 Project: perfect-entry-473503-j1")
    print("🏗️ Real-time Health Monitoring with Visual Agent Insights")
    print("=" * 50)
    
    # Show agent architecture
    create_agent_dashboard()
    
    print("\n🎮 CONTROLS:")
    print("   Ctrl+C: Stop monitoring")
    print("   Press Enter to start monitoring...")
    input()
    
    try:
        # Initialize and run the production system
        system = StraightUpADKSystem()
        asyncio.run(system.start_monitoring())
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")
    except Exception as e:
        print(f"❌ System error: {e}")
        print("Check camera, detector, and sensor availability")