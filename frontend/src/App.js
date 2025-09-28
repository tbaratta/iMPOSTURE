import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  Container,
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  Button,
  Alert,
  CircularProgress,
  Paper
} from '@mui/material';
import {
  Line
} from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const API_BASE = 'http://localhost:5000';

function App() {
  const [healthSummary, setHealthSummary] = useState(null);
  const [realtimeStatus, setRealtimeStatus] = useState(null);
  const [chartData, setChartData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [autoRefresh, setAutoRefresh] = useState(true);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      setError(null);

      const [summaryRes, realtimeRes, chartRes] = await Promise.all([
        axios.get(`${API_BASE}/api/health/summary`),
        axios.get(`${API_BASE}/api/health/realtime`),
        axios.get(`${API_BASE}/api/charts/focus-trend`)
      ]);

      setHealthSummary(summaryRes.data);
      setRealtimeStatus(realtimeRes.data);
      setChartData(chartRes.data);
    } catch (err) {
      setError('Failed to load dashboard data: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
  }, []);

  useEffect(() => {
    let interval;
    if (autoRefresh) {
      interval = setInterval(fetchDashboardData, 30000); // 30 seconds
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [autoRefresh]);

  const getScoreColor = (score) => {
    if (score > 0.7) return 'success';
    if (score > 0.4) return 'warning';
    return 'error';
  };

  const getGradeColor = (grade) => {
    const colors = {
      'A': '#38a169',
      'B': '#3182ce', 
      'C': '#d69e2e',
      'D': '#e53e3e',
      'F': '#742a2a'
    };
    return colors[grade] || '#666';
  };

  const getSystemStatusChip = () => {
    if (!realtimeStatus || realtimeStatus.status !== 'success') {
      return <Chip label="🔴 System Offline" color="error" />;
    }
    
    if (realtimeStatus.is_recent) {
      return <Chip label="🟢 System Active" color="success" />;
    }
    
    return <Chip label={`🟡 Last seen ${Math.round(realtimeStatus.last_update_minutes_ago)}m ago`} color="warning" />;
  };

  if (loading && !healthSummary) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4, textAlign: 'center' }}>
        <CircularProgress size={60} />
        <Typography variant="h6" sx={{ mt: 2 }}>
          Loading StraightUp Dashboard...
        </Typography>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      {/* Header */}
      <Paper sx={{ p: 3, mb: 3, background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', color: 'white' }}>
        <Typography variant="h3" component="h1" gutterBottom align="center">
          🎯 StraightUp Health Dashboard
        </Typography>
        <Typography variant="h6" align="center" sx={{ opacity: 0.9 }}>
          Real-time wellness monitoring from Google ADK system
        </Typography>
        <Typography variant="body1" align="center" sx={{ mt: 1 }}>
          <strong>Project:</strong> perfect-entry-473503-j1
        </Typography>
        
        <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2, mt: 2 }}>
          {getSystemStatusChip()}
          <Button 
            variant="contained" 
            onClick={fetchDashboardData}
            disabled={loading}
            sx={{ bgcolor: 'rgba(255,255,255,0.2)' }}
          >
            🔄 Refresh
          </Button>
          <Button 
            variant="contained"
            onClick={() => setAutoRefresh(!autoRefresh)}
            sx={{ bgcolor: 'rgba(255,255,255,0.2)' }}
          >
            ⏰ Auto Refresh: {autoRefresh ? 'ON' : 'OFF'}
          </Button>
        </Box>
      </Paper>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {healthSummary && healthSummary.status === 'success' && (
        <>
          {/* Summary Cards */}
          <Grid container spacing={3} sx={{ mb: 3 }}>
            {/* Overall Health Grade */}
            <Grid item xs={12} md={3}>
              <Card sx={{ height: '100%' }}>
                <CardContent sx={{ textAlign: 'center' }}>
                  <Typography variant="h6" gutterBottom>
                    📊 Overall Health Grade
                  </Typography>
                  <Box
                    sx={{
                      width: 100,
                      height: 100,
                      borderRadius: '50%',
                      background: getGradeColor(healthSummary.health_grade),
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      margin: '0 auto 16px',
                      color: 'white',
                      fontSize: '3rem',
                      fontWeight: 'bold'
                    }}
                  >
                    {healthSummary.health_grade}
                  </Box>
                  <Typography variant="body2">
                    Data Points: {healthSummary.data_points}
                  </Typography>
                  <Typography variant="body2">
                    Last Updated: {new Date(healthSummary.last_updated).toLocaleString()}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            {/* Focus & Posture */}
            <Grid item xs={12} md={3}>
              <Card sx={{ height: '100%' }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    🎯 Focus & Posture
                  </Typography>
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2">Focus Score</Typography>
                    <Chip 
                      label={`${(healthSummary.averages.focus_score * 100).toFixed(1)}%`}
                      color={getScoreColor(healthSummary.averages.focus_score)}
                      sx={{ fontWeight: 'bold' }}
                    />
                  </Box>
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2">Posture Score</Typography>
                    <Chip 
                      label={`${(healthSummary.averages.posture_score * 100).toFixed(1)}%`}
                      color={getScoreColor(healthSummary.averages.posture_score)}
                      sx={{ fontWeight: 'bold' }}
                    />
                  </Box>
                  <Box>
                    <Typography variant="body2">Trend</Typography>
                    <Typography variant="body1">
                      {healthSummary.trends.focus_trend === 'improving' ? '📈' : 
                       healthSummary.trends.focus_trend === 'declining' ? '📉' : '➡️'} {healthSummary.trends.focus_trend}
                    </Typography>
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            {/* Phone Usage */}
            <Grid item xs={12} md={3}>
              <Card sx={{ height: '100%' }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    📱 Phone Usage
                  </Typography>
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2">Total Today</Typography>
                    <Typography variant="h4" color="primary">
                      {healthSummary.totals.phone_usage_minutes.toFixed(1)} min
                    </Typography>
                  </Box>
                  <Box>
                    <Typography variant="body2">Status</Typography>
                    <Chip 
                      label={healthSummary.totals.phone_usage_minutes < 10 ? "✅ Good discipline" : 
                            healthSummary.totals.phone_usage_minutes < 30 ? "⚠️ Moderate usage" : "🔴 High usage"}
                      color={healthSummary.totals.phone_usage_minutes < 10 ? "success" : 
                            healthSummary.totals.phone_usage_minutes < 30 ? "warning" : "error"}
                    />
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            {/* Environment */}
            <Grid item xs={12} md={3}>
              <Card sx={{ height: '100%' }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    🔊 Environment
                  </Typography>
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2">Noise Level</Typography>
                    <Chip 
                      label={`${(healthSummary.averages.noise_level * 100).toFixed(1)}%`}
                      color={getScoreColor(1 - healthSummary.averages.noise_level)}
                      sx={{ fontWeight: 'bold' }}
                    />
                  </Box>
                  <Box>
                    <Typography variant="body2">Classification</Typography>
                    <Typography variant="body1">
                      {healthSummary.averages.noise_level < 0.3 ? "🟢 Quiet" :
                       healthSummary.averages.noise_level < 0.6 ? "🟡 Moderate" : "🔴 Noisy"}
                    </Typography>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          {/* Charts */}
          {chartData && (
            <Paper sx={{ p: 3, mb: 3 }}>
              <Typography variant="h6" gutterBottom>
                📈 Focus & Posture Trends (Last 12 Hours)
              </Typography>
              <Box sx={{ height: 400 }}>
                <Line 
                  data={chartData}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                      y: {
                        beginAtZero: true,
                        max: 1,
                        title: {
                          display: true,
                          text: 'Score (0-1)'
                        }
                      },
                      x: {
                        title: {
                          display: true,
                          text: 'Time'
                        }
                      }
                    },
                    plugins: {
                      legend: {
                        display: true,
                        position: 'top'
                      }
                    }
                  }}
                />
              </Box>
            </Paper>
          )}

          {/* Recommendations */}
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                💡 Top Recommendations
              </Typography>
              {healthSummary.top_recommendations && healthSummary.top_recommendations.length > 0 ? (
                <Box>
                  {healthSummary.top_recommendations.slice(0, 5).map((rec, index) => (
                    <Alert key={index} severity="info" sx={{ mb: 1 }}>
                      {rec.text} <strong>({rec.count}x)</strong>
                    </Alert>
                  ))}
                </Box>
              ) : (
                <Alert severity="success">
                  No specific recommendations - you're doing great! 🌟
                </Alert>
              )}
            </CardContent>
          </Card>
        </>
      )}

      {healthSummary && healthSummary.status !== 'success' && (
        <Alert severity="warning">
          No health data available. Make sure the ADK production system is running.
        </Alert>
      )}
    </Container>
  );
}

export default App;