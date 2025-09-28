# 🖥️ StraightUp Desktop Dashboard

A native desktop application for real-time health monitoring that connects directly to your Google ADK production system.

## 📊 **Desktop App Overview**

The desktop dashboard provides a native Windows application that pulls real-time health data directly from your Google Cloud Logging:

- **Real-time health metrics** from your ADK agents
- **Interactive charts and trends** showing focus and posture over time  
- **Health recommendations** based on actual usage patterns
- **System status monitoring** to track ADK agent activity
- **Overall health grading** with A-F scoring system
- **Native desktop experience** with no browser required

## 🔄 **Data Flow Architecture**

```
ADK Production System → Google Cloud Logging → Desktop App
```

1. **ADK Production System** (`../backend/adk_production.py`) sends structured data to Google Cloud
2. **Desktop App** queries Google Cloud Logging directly for real-time data
3. **Native UI** displays the data in beautiful charts and metrics with auto-refresh

## 🚀 **Quick Start**

### One-Click Setup (Recommended)

```powershell
# Navigate to frontend directory
cd frontend

# Run the smart launcher - automatically installs dependencies and starts app
python setup_desktop_launcher.py
```

This will:
- ✅ Check Python compatibility 
- 📦 Install required dependencies (matplotlib, google-cloud-logging, customtkinter)
- 🎨 Detect available UI frameworks
- 🚀 Launch the best available desktop app

### Manual Launch Options

#### Option 1: Modern UI (CustomTkinter)
```powershell
python desktop_app.py
```

#### Option 2: Classic UI (Pure Tkinter)  
```powershell
python desktop_tkinter.py
```

## 🖥️ **Desktop App Features**

### 📊 **Dashboard Tab**
- **Health Grade Card**: Overall A-F scoring with data point count
- **Focus & Posture Card**: Real-time scores with trend indicators (📈📉➡️)
- **Phone Usage Card**: Daily totals and usage status
- **Environment Card**: Noise levels and classifications

### 📈 **Charts Tab**
- **Interactive Time Ranges**: 1 hour, 6 hours, 12 hours, 24 hours
- **Focus & Posture Trends**: Beautiful line charts with real data
- **Real-time Updates**: Charts refresh with new data automatically
- **Matplotlib Integration**: Professional quality visualizations

### 💡 **Recommendations Tab**
- **Top Health Recommendations**: Most frequent suggestions from ADK agents
- **Frequency Tracking**: Shows how often each recommendation appears (count)
- **Scrollable Interface**: View all recommendations with clean layout
- **Actionable Insights**: Specific posture and wellness guidance

### ⚙️ **Settings Tab**
- **Auto-Refresh Controls**: 10 seconds, 30 seconds, 60 seconds, 5 minutes
- **Google Cloud Status**: Real-time connection monitoring
- **App Information**: Version details and feature overview
- **Theme Support**: Light/dark themes (CustomTkinter version)

## 🎯 **App Comparison**

| Feature | Modern UI (CustomTkinter) | Classic UI (Tkinter) |
|---------|--------------------------|---------------------|
| **Interface** | 🎨 Modern, sleek design | 🔧 Traditional Windows UI |
| **Themes** | ✅ Light/Dark themes | ❌ System default only |
| **Dependencies** | CustomTkinter, matplotlib | Pure Tkinter, matplotlib |
| **Performance** | High (GPU accelerated) | High (lightweight) |
| **Compatibility** | Python 3.8+ | Python 3.6+ |
| **Reliability** | Modern framework | Battle-tested standard |

## 🌐 **Google Cloud Integration**

The desktop apps connect directly to your Google Cloud Logging:

- **Project ID**: `perfect-entry-473503-j1`
- **Logger Name**: `straightup-adk-production`
- **Data Source**: `adk_production_system`
- **Direct Connection**: No web server required

### Real-time Data Structure
Your ADK system sends this structured data:

```json
{
  "cycle": 123,
  "timestamp": "2025-09-28T12:34:56Z",
  "focus_score": 0.75,
  "posture_score": 0.65,
  "phone_usage_seconds": 12.5,
  "noise_level": 0.25,
  "recommendations": ["🎯 Focus on alignment", "🟡 Minor shoulder imbalance"],
  "project_id": "perfect-entry-473503-j1",
  "source": "adk_production_system",
  "agent_status": "operational"
}
```

## 🛠 **Technical Details**

### Desktop App Architecture
- **Google Cloud Logging Client**: Direct queries to your logged health data
- **Background Threading**: Non-blocking data fetching for smooth UI
- **Auto-refresh Timer**: Configurable real-time updates (10s - 5min)
- **Error Handling**: Graceful fallbacks with sample data when offline

### Dependencies
```bash
# Required packages (auto-installed by launcher)
matplotlib>=3.5.0          # Charts and plotting
google-cloud-logging>=3.0  # Google Cloud integration

# Optional packages  
customtkinter>=5.0         # Modern UI framework (recommended)
```

### System Requirements
- **OS**: Windows 10/11 (primary), Linux/macOS (compatible)
- **Python**: 3.8+ (3.11+ recommended)
- **RAM**: 200MB minimum for app
- **Network**: Internet connection for Google Cloud data

## 🎮 **Usage Examples**

### Daily Health Monitoring
1. **Start ADK System**: Run `python ../backend/adk_production.py`
2. **Launch Desktop App**: Run `python setup_desktop_launcher.py`
3. **Monitor Real-time**: Watch health metrics update every 30 seconds
4. **Check Trends**: Switch to Charts tab for historical analysis

### Review Health Insights
- **Dashboard Overview**: Quick health grade and current status
- **Trend Analysis**: View focus/posture changes over different time ranges
- **Recommendations**: See what improvements your ADK system suggests
- **System Status**: Confirm your ADK system is actively monitoring

## 🚨 **Troubleshooting**

### Desktop App Won't Start
1. **Python Version**: Ensure Python 3.8+ is installed
2. **Dependencies**: Run launcher to auto-install packages
3. **Firewall**: Allow Python through Windows firewall for Google Cloud access

### No Data Showing  
1. **ADK System**: Ensure `../backend/adk_production.py` is running
2. **Google Cloud**: Check data in Cloud Console logs
3. **Project ID**: Verify `perfect-entry-473503-j1` is correct
4. **Credentials**: Ensure Google Cloud authentication is set up

### Charts Not Updating
1. **Auto-refresh**: Check if auto-refresh is enabled (⏰ button)
2. **Time Range**: Try different time ranges if recent data is limited
3. **Data Availability**: Verify ADK system has been running for selected period

### UI Issues
- **Modern UI Problems**: Try Classic UI with `python desktop_tkinter.py`
- **Display Scaling**: Adjust Windows display scaling if UI elements overlap
- **Theme Issues**: Use Settings tab to switch themes (Modern UI only)

## 🌟 **Advantages of Desktop App**

### vs Web Dashboard
- ✅ **No Browser Required**: Native desktop experience
- ✅ **Direct Google Cloud Connection**: No web server middleman
- ✅ **Better Performance**: Native rendering and caching
- ✅ **Offline Resilience**: Graceful handling of network issues
- ✅ **System Integration**: Native Windows notifications and taskbar

### vs Command Line
- ✅ **Visual Interface**: Beautiful charts and real-time updates
- ✅ **User-Friendly**: No command line knowledge required
- ✅ **Interactive**: Click and explore your health data
- ✅ **Persistent**: Runs continuously in background

## 📞 **Support**

### Quick Diagnostics
1. **Run Launcher**: `python setup_desktop_launcher.py` shows setup status
2. **Check Logs**: Desktop app prints status to console window
3. **Google Cloud Console**: Verify data at https://console.cloud.google.com/logs
4. **Test Connection**: Settings tab shows Google Cloud connection status

### Common Solutions
- **Dependency Issues**: Re-run launcher to reinstall packages
- **Authentication**: Ensure `GOOGLE_CLOUD_PROJECT` environment variable is set
- **Performance**: Close other resource-intensive applications
- **Updates**: Pull latest code for bug fixes and improvements

The desktop dashboard provides the best experience for monitoring your StraightUp health data! 🎯✨