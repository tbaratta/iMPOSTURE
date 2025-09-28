"""
StraightUp Dashboard Setup Script
Installs dependencies and starts the dashboard server
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def main():
    print("🎯 StraightUp Dashboard Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('dashboard_api.py'):
        print("❌ Please run this script from the frontend directory")
        return
    
    print("📦 Installing Python dependencies...")
    
    # Install Flask and dependencies
    dependencies = [
        "flask",
        "flask-cors", 
        "google-cloud-logging"
    ]
    
    for dep in dependencies:
        if not run_command(f"pip install {dep}", f"Installing {dep}"):
            print(f"⚠️ Failed to install {dep}, but continuing...")
    
    print("\n🚀 Starting Dashboard API Server...")
    print("🌐 Dashboard will be available at: http://localhost:5000")
    print("📊 API endpoints:")
    print("   - /api/health/summary")
    print("   - /api/health/realtime") 
    print("   - /api/charts/focus-trend")
    print("\n⚡ Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Start the Flask server
        subprocess.run([sys.executable, "dashboard_api.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Dashboard server stopped")
    except Exception as e:
        print(f"❌ Server error: {e}")

if __name__ == "__main__":
    main()