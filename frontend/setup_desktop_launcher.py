"""
StraightUp Desktop Setup - Choose Your Interface
Install dependencies and launch the desktop health dashboard
Project: perfect-entry-473503-j1
"""

import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """Check Python version compatibility"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version {sys.version.split()[0]} is compatible")
    return True

def install_package(package_name, description=""):
    """Install a Python package using pip"""
    try:
        print(f"📦 Installing {package_name}...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", package_name
        ], capture_output=True, text=True, check=True)
        print(f"✅ {package_name} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package_name}: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_and_install_dependencies():
    """Check and install required dependencies"""
    print("🔍 Checking dependencies...")
    
    # Required packages
    base_packages = [
        ("requests", "HTTP client for API calls"),
        ("matplotlib", "Charts and plotting"),
        ("google-cloud-logging", "Google Cloud integration")
    ]
    
    optional_packages = [
        ("customtkinter", "Modern UI framework (optional)")
    ]
    
    # Install base packages
    all_success = True
    for package, desc in base_packages:
        if not install_package(package, desc):
            all_success = False
    
    # Try to install customtkinter (optional)
    customtkinter_available = install_package("customtkinter", "Modern UI framework")
    
    return all_success, customtkinter_available

def launch_app(app_choice, customtkinter_available):
    """Launch the selected desktop app"""
    current_dir = Path(__file__).parent
    
    # App choices
    if app_choice == "1" and customtkinter_available:
        app_file = current_dir / "modern_desktop_app.py"
        app_name = "Modern CustomTkinter (Web UI Style)"
    elif app_choice == "2":
        app_file = current_dir / "modern_tkinter_app.py"
        app_name = "Modern Pure Tkinter (Web UI Style)"
    elif app_choice == "3" and customtkinter_available:
        app_file = current_dir / "desktop_app.py"
        app_name = "Original CustomTkinter Dashboard"
    elif app_choice == "4":
        app_file = current_dir / "desktop_tkinter.py"
        app_name = "Original Pure Tkinter Dashboard"
    else:
        # Default fallback
        if customtkinter_available:
            app_file = current_dir / "modern_desktop_app.py"
            app_name = "Modern CustomTkinter (Web UI Style)"
        else:
            app_file = current_dir / "modern_tkinter_app.py"
            app_name = "Modern Pure Tkinter (Web UI Style)"
    
    if not app_file.exists():
        print(f"❌ Desktop app file not found: {app_file}")
        return False
    
    print(f"🚀 Launching {app_name} desktop app...")
    print(f"📁 App file: {app_file}")
    
    try:
        # Launch the desktop app
        subprocess.run([sys.executable, str(app_file)], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch desktop app: {e}")
        return False
    except KeyboardInterrupt:
        print("\n🛑 App launch cancelled by user")
        return False

def main():
    """Main setup function"""
    print("🖥️ StraightUp Desktop Health Dashboard Setup")
    print("=" * 60)
    print("🎯 Project: perfect-entry-473503-j1")
    print("📊 Real-time health monitoring from Google ADK system")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Install dependencies
    success, customtkinter_available = check_and_install_dependencies()
    
    if not success:
        print("\n❌ Some required dependencies failed to install")
        print("Please check the error messages above and try again")
        return
    
    print("\n✅ All dependencies checked/installed successfully!")
    
    # Choose desktop app version
    print("\n🖥️ Choose your desktop interface:")
    print("=" * 50)
    
    if customtkinter_available:
        print("1. 🎨 Modern Web UI Style (CustomTkinter) - NEW!")
        print("   • Beautiful web UI design in desktop")
        print("   • Live wellness chips and progress bars")
        print("   • Session timer with modern controls")
        print()
    
    print("2. 🎨 Modern Web UI Style (Pure Tkinter) - NEW!")
    print("   • Same beautiful design, no dependencies")
    print("   • Web UI style with dark theme")
    print("   • Works without CustomTkinter")
    print()
    
    if customtkinter_available:
        print("3. 📊 Original Dashboard (CustomTkinter)")
        print("   • Full dashboard with charts")
        print("   • Health analytics")
        print("   • Traditional desktop layout")
        print()
    
    print("4. � Original Dashboard (Pure Tkinter)")
    print("   • Traditional dashboard interface")
    print("   • Charts and detailed metrics")
    print("   • No external GUI dependencies")
    print()
    
    if customtkinter_available:
        choice = input("Select interface (1-4, default=1): ").strip() or "1"
    else:
        print("⚠️ CustomTkinter not available, showing options 2 & 4")
        choice = input("Select interface (2 or 4, default=2): ").strip() or "2"
    
    # Launch the app
    print("\n🚀 Starting desktop application...")
    
    if launch_app(choice, customtkinter_available):
        print("✅ Desktop app launched successfully!")
    else:
        print("❌ Failed to launch desktop app")
    
    print("\nPress Enter to exit...")
    input()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Setup cancelled by user")
    except Exception as e:
        print(f"❌ Setup error: {e}")
        print("\nPress Enter to exit...")
        input()