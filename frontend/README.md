# 🎯 StraightUp Frontend Dashboard

A comprehensive frontend dashboard that pulls real-time health data from your Google ADK production system.

## 📊 **Dashboard Overview**

The frontend dashboard connects to your Google Cloud Logging data to provide:

- **Real-time health metrics** from your ADK agents
- **Visual charts and trends** showing focus and posture over time  
- **Health recommendations** based on actual usage patterns
- **System status monitoring** to track ADK agent activity
- **Overall health grading** with A-F scoring system

## 🔄 **Data Flow Architecture**

```
ADK Production System → Google Cloud Logging → Dashboard API → Frontend UI
```

1. **ADK Production System** (`adk_production.py`) sends structured data to Google Cloud
2. **Dashboard API** (`dashboard_api.py`) queries Google Cloud Logging for data
3. **Frontend UI** displays the data in beautiful charts and metrics

## 🚀 **Quick Start**

### Option 1: Flask + HTML Dashboard (Recommended)

```powershell
# Navigate to frontend directory
cd frontend

# Install and run the dashboard
python setup_dashboard.py
```

The dashboard will be available at: `http://localhost:5000`

### Option 2: React Dashboard (Advanced)

```powershell
# Install Node.js dependencies
npm install

# Start React development server
npm start
```

## 📡 **API Endpoints**

The dashboard API provides several endpoints for accessing your health data:

| Endpoint | Description | Example Response |
|----------|-------------|------------------|
| `/api/health/summary` | Aggregated health statistics | Focus trends, averages, recommendations |
| `/api/health/realtime` | Current system status | Last update time, current metrics |
| `/api/health/recent` | Raw health data points | Individual cycles with full metrics |
| `/api/charts/focus-trend` | Chart-ready data | Time series for focus/posture |

## 🎯 **Data Structure**

Your ADK system sends this structured data to Google Cloud:

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

## 🌐 **Google Cloud Integration**

The dashboard pulls data from your Google Cloud Logging:

- **Project ID**: `perfect-entry-473503-j1`
- **Logger Name**: `straightup-adk-production`
- **Data Source**: `adk_production_system`

### Viewing Raw Data in Google Cloud Console

You can also view your data directly in Google Cloud:

```
https://console.cloud.google.com/logs/query?project=perfect-entry-473503-j1
```

Use this query to see your health data:
```
logName="projects/perfect-entry-473503-j1/logs/straightup-adk-production"
jsonPayload.source="adk_production_system"
```

## 📊 **Dashboard Features**

### 🎯 **Health Overview Cards**
- **Overall Health Grade**: A-F scoring based on all metrics
- **Focus & Posture**: Real-time scores with trend indicators
- **Phone Usage**: Daily totals and session tracking
- **Environment**: Noise levels and classifications

### 📈 **Interactive Charts**
- **Focus & Posture Trends**: Line charts showing 12-hour history
- **Real-time Updates**: Auto-refresh every 30 seconds
- **Responsive Design**: Works on desktop and mobile

### 💡 **Smart Recommendations**
- **Top Recommendations**: Most frequent suggestions from ADK agents
- **Frequency Tracking**: Shows how often each recommendation appears
- **Actionable Insights**: Specific posture and wellness guidance

### 🔄 **System Monitoring**
- **Real-time Status**: Shows if ADK system is actively running
- **Last Update Time**: When data was last received
- **Connection Health**: Visual indicators for system status

## 🛠 **Technical Details**

### Flask Backend (`dashboard_api.py`)
- **Google Cloud Logging Client**: Queries your logged health data
- **REST API**: Provides JSON endpoints for frontend consumption
- **CORS Enabled**: Allows frontend access from different origins
- **Error Handling**: Graceful fallbacks when Cloud data unavailable

### Frontend Options

#### HTML/JavaScript Dashboard
- **Pure HTML/CSS/JS**: No build process required
- **Chart.js Integration**: Beautiful responsive charts
- **Material Design**: Clean, professional interface
- **Auto-refresh**: Configurable real-time updates

#### React Dashboard
- **Material-UI Components**: Professional UI components
- **Axios HTTP Client**: Reliable API communication  
- **Chart.js Integration**: Interactive data visualization
- **TypeScript Ready**: Full type safety support

## 🔧 **Configuration**

### Environment Variables
```bash
GOOGLE_CLOUD_PROJECT=perfect-entry-473503-j1
```

### API Configuration
```python
# In dashboard_api.py
app.run(host='0.0.0.0', port=5000, debug=True)
```

### Frontend Configuration
```javascript
// In src/App.js
const API_BASE = 'http://localhost:5000';
```

## 🎮 **Usage Examples**

### Monitor Real-time Health
1. Start your ADK production system: `python adk_production.py`
2. Start the dashboard: `python setup_dashboard.py`
3. Open browser to: `http://localhost:5000`
4. Watch real-time health metrics update every 30 seconds

### Check Historical Trends
- View focus and posture trends over the last 12 hours
- See which recommendations appear most frequently
- Track overall health grade improvements

### System Status Monitoring
- Green indicator: ADK system actively sending data
- Yellow indicator: Data received recently but not currently active
- Red indicator: No recent data, system may be offline

## 🚨 **Troubleshooting**

### No Data Showing
1. **Check ADK System**: Ensure `adk_production.py` is running
2. **Verify Google Cloud**: Confirm data in Cloud Logging console
3. **Check Project ID**: Ensure `perfect-entry-473503-j1` is correct
4. **Network Issues**: Verify API endpoints are accessible

### Dashboard Not Loading
1. **Install Dependencies**: Run `pip install flask flask-cors google-cloud-logging`
2. **Port Conflicts**: Change port in `dashboard_api.py` if 5000 is busy
3. **CORS Issues**: Ensure `flask-cors` is installed and configured

### Chart Not Updating
1. **Auto-refresh**: Check if auto-refresh is enabled
2. **Data Format**: Verify API returns proper chart data structure
3. **Browser Cache**: Hard refresh or clear browser cache

## 🌟 **Future Enhancements**

Potential improvements for the dashboard:

- **Real-time WebSocket Updates**: Live data streaming instead of polling
- **Historical Data Export**: Download health data as CSV/JSON
- **Custom Alert Thresholds**: Set personalized health targets
- **Multi-user Support**: Track multiple users in same dashboard
- **Mobile App**: Native iOS/Android dashboard application
- **Advanced Analytics**: ML-powered health insights and predictions

## 📞 **Support**

For issues with the dashboard:

1. **Check Google Cloud Console**: Verify data is being logged
2. **Review API Logs**: Check dashboard_api.py console output
3. **Test Endpoints**: Manually test API endpoints in browser
4. **Verify Dependencies**: Ensure all Python packages installed

The dashboard is designed to work seamlessly with your existing ADK production system!