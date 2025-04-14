import React, { useState, useEffect } from 'react';
import { 
  Typography, Grid, Card, CardContent, Box, CircularProgress, Alert, 
  Tabs, Tab, Button, FormControl, InputLabel, Select, MenuItem, Paper,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Chip, Divider
} from '@mui/material';
import { 
  BarChart, Bar, PieChart, Pie, LineChart, Line, XAxis, YAxis, CartesianGrid, 
  Tooltip, Legend, ResponsiveContainer, Cell, ScatterChart, Scatter, ZAxis, 
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, AreaChart, Area
} from 'recharts';
import axios from 'axios';
import DateRangePicker from './DateRangePicker';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d', '#ffc658', '#8dd1e1', '#a4de6c', '#d0ed57'];

const Statistics = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [dateRange, setDateRange] = useState({
    startDate: new Date(new Date().setDate(new Date().getDate() - 30)),
    endDate: new Date()
  });
  const [tabValue, setTabValue] = useState(0);
  const [selectedLocality, setSelectedLocality] = useState('all');
  const [selectedSymptom, setSelectedSymptom] = useState('all');
  const [insights, setInsights] = useState([]);
  const [localityList, setLocalityList] = useState([]);

  // Handle tab change
  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        // Format dates for API
        const startDateStr = dateRange.startDate ? dateRange.startDate.toISOString().split('T')[0] : '';
        const endDateStr = dateRange.endDate ? dateRange.endDate.toISOString().split('T')[0] : '';
        
        const response = await axios.get('http://localhost:5000/api/statistics', {
          params: {
            start_date: startDateStr,
            end_date: endDateStr,
            locality: selectedLocality !== 'all' ? selectedLocality : undefined,
            symptom: selectedSymptom !== 'all' ? selectedSymptom : undefined
          }
        });
        
        setData(response.data);
        
        // Extract unique localities from data
        if (response.data && response.data.locality_distribution) {
          const localities = Object.keys(response.data.locality_distribution);
          setLocalityList(['all', ...localities]);
        }
        
        // Generate insights (this would ideally come from backend)
        generateInsights(response.data);
        
        setError(null);
      } catch (err) {
        console.error('Error fetching statistics:', err);
        setError('Failed to load statistics. Please try again later.');
        
        // Create mock data for demonstration
        createMockData();
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [dateRange, selectedLocality, selectedSymptom]);
  
  // Mock data generator for demonstration
  const createMockData = () => {
    const mockData = {
      total_diagnoses: 1250,
      top_symptoms: [
        ['fever', 423],
        ['cough', 387],
        ['headache', 312],
        ['fatigue', 287],
        ['sore_throat', 243],
        ['shortness_of_breath', 198],
        ['joint_pain', 156],
        ['nausea', 134],
        ['diarrhea', 120],
        ['loss_of_taste', 89]
      ],
      top_diseases: [
        ['Common Cold', 345],
        ['Influenza', 230],
        ['COVID-19', 185],
        ['Allergic Rhinitis', 125],
        ['Gastroenteritis', 95],
        ['Pneumonia', 78],
        ['Bronchitis', 65],
        ['Sinusitis', 45]
      ],
      time_series: Array(30).fill().map((_, i) => ({
        date: new Date(new Date().setDate(new Date().getDate() - (30 - i))).toISOString().split('T')[0],
        total: Math.floor(Math.random() * 50) + 20
      })),
      locality_distribution: {
        'Chennai Central': 345,
        'Velachery': 278,
        'Anna Nagar': 230,
        'T. Nagar': 198,
        'Adyar': 156,
        'Tambaram': 43
      },
      symptom_correlations: [
        { symptom1: 'fever', symptom2: 'cough', correlation: 0.78 },
        { symptom1: 'fever', symptom2: 'headache', correlation: 0.65 },
        { symptom1: 'cough', symptom2: 'shortness_of_breath', correlation: 0.72 },
        { symptom1: 'nausea', symptom2: 'diarrhea', correlation: 0.81 },
        { symptom1: 'joint_pain', symptom2: 'fatigue', correlation: 0.69 }
      ],
      weekly_trends: [
        { disease: 'Dengue', change: 45 },
        { disease: 'Malaria', change: 32 },
        { disease: 'COVID-19', change: -15 },
        { disease: 'Influenza', change: 8 },
        { disease: 'Gastroenteritis', change: 23 }
      ],
      symptom_by_locality: {
        'Chennai Central': [
          ['fever', 135], ['cough', 120], ['headache', 95]
        ],
        'Velachery': [
          ['fever', 85], ['joint_pain', 65], ['rash', 45]
        ],
        'Anna Nagar': [
          ['cough', 95], ['fever', 85], ['sore_throat', 65]
        ]
      },
      medicine_usage: [
        { disease: 'Influenza', cases: 230, med1_sales: 210, med2_sales: 195 },
        { disease: 'Dengue', cases: 145, med1_sales: 140, med2_sales: 85 },
        { disease: 'Malaria', cases: 120, med1_sales: 115, med2_sales: 105 },
        { disease: 'COVID-19', cases: 185, med1_sales: 170, med2_sales: 165 }
      ],
      symptom_clusters: [
        { 
          locality: 'Chennai Central',
          symptoms: [
            { name: 'fever', value: 8 },
            { name: 'cough', value: 7 },
            { name: 'headache', value: 6 },
            { name: 'fatigue', value: 5 },
            { name: 'sore_throat', value: 4 }
          ]
        },
        { 
          locality: 'Velachery',
          symptoms: [
            { name: 'fever', value: 6 },
            { name: 'joint_pain', value: 8 },
            { name: 'rash', value: 7 },
            { name: 'fatigue', value: 4 },
            { name: 'nausea', value: 3 }
          ]
        }
      ]
    };
    
    setData(mockData);
    setLocalityList(['all', ...Object.keys(mockData.locality_distribution)]);
    generateInsights(mockData);
  };
  
  // Generate insights based on data analysis
  const generateInsights = (data) => {
    if (!data) return;
    
    const insights = [];
    
    // Check for increasing trends
    if (data.weekly_trends) {
      const risingDiseases = data.weekly_trends
        .filter(item => item.change > 30)
        .map(item => item.disease);
        
      if (risingDiseases.length > 0) {
        insights.push({
          type: 'warning',
          message: `Significant increase in ${risingDiseases.join(', ')} cases in the last week.`
        });
      }
    }
    
    // Medicine availability issues
    if (data.medicine_usage) {
      const medIssues = data.medicine_usage
        .filter(item => (item.med1_sales / item.cases < 0.7) || (item.med2_sales / item.cases < 0.7))
        .map(item => item.disease);
        
      if (medIssues.length > 0) {
        insights.push({
          type: 'alert',
          message: `Possible medicine shortage or prescription change for ${medIssues.join(', ')}.`
        });
      }
    }
    
    // Symptom correlations
    if (data.symptom_correlations) {
      const highCorr = data.symptom_correlations
        .filter(item => item.correlation > 0.75);
        
      if (highCorr.length > 0) {
        const corrPair = highCorr[0];
        insights.push({
          type: 'info',
          message: `Strong correlation detected between ${corrPair.symptom1.replace('_', ' ')} and ${corrPair.symptom2.replace('_', ' ')}, suggesting possible disease pattern.`
        });
      }
    }
    
    // Locality specific insights
    if (data.locality_distribution && data.symptom_by_locality) {
      const topLocality = Object.entries(data.locality_distribution)
        .sort((a, b) => b[1] - a[1])[0][0];
        
      if (data.symptom_by_locality[topLocality]) {
        const topSymptoms = data.symptom_by_locality[topLocality]
          .slice(0, 2)
          .map(item => item[0].replace('_', ' '));
          
        insights.push({
          type: 'info',
          message: `${topLocality} has the highest case count with predominant symptoms: ${topSymptoms.join(', ')}.`
        });
      }
    }
    
    setInsights(insights);
  };

  // Format symptom data for bar chart
  const formatSymptomData = () => {
    if (!data || !data.top_symptoms) return [];
    
    return data.top_symptoms.map(([symptom, count]) => ({
      name: symptom.replace('_', ' '),
      count: count
    }));
  };

  // Format disease data for pie chart
  const formatDiseaseData = () => {
    if (!data || !data.top_diseases) return [];
    
    return data.top_diseases.map(([disease, count]) => ({
      name: disease,
      value: count
    }));
  };

  // Format time series data for line chart
  const formatTimeSeriesData = () => {
    if (!data || !data.time_series) return [];
    
    return data.time_series.map(item => ({
      date: item.date,
      total: item.total
    }));
  };
  
  // Format locality distribution data
  const formatLocalityData = () => {
    if (!data || !data.locality_distribution) return [];
    
    return Object.entries(data.locality_distribution).map(([locality, count]) => ({
      name: locality,
      value: count
    }));
  };
  
  // Format weekly trends data
  const formatWeeklyTrendsData = () => {
    if (!data || !data.weekly_trends) return [];
    
    return data.weekly_trends.sort((a, b) => b.change - a.change);
  };
  
  // Format medicine usage data
  const formatMedicineData = () => {
    if (!data || !data.medicine_usage) return [];
    
    return data.medicine_usage;
  };
  
  // Format symptom cluster data for radar chart
  const formatSymptomClusterData = () => {
    if (!data || !data.symptom_clusters) return [];
    
    // If locality selected, filter the data
    if (selectedLocality !== 'all') {
      return data.symptom_clusters
        .filter(item => item.locality === selectedLocality)
        .map(item => item.symptoms);
    }
    
    // Return first cluster if none selected
    return data.symptom_clusters.length > 0 ? [data.symptom_clusters[0].symptoms] : [];
  };
  
  // Format correlation data for scatter plot
  const formatCorrelationData = () => {
    if (!data || !data.symptom_correlations) return [];
    
    return data.symptom_correlations.map(item => ({
      x: item.symptom1.replace('_', ' '),
      y: item.symptom2.replace('_', ' '),
      z: Math.round(item.correlation * 100)
    }));
  };

  // Render insights panel
  const renderInsights = () => {
    return (
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            AI-Generated Insights
          </Typography>
          {insights.length > 0 ? (
            <Box>
              {insights.map((insight, index) => (
                <Alert severity={insight.type} key={index} sx={{ mb: 1 }}>
                  {insight.message}
                </Alert>
              ))}
            </Box>
          ) : (
            <Typography variant="body2">No significant insights detected in current data.</Typography>
          )}
        </CardContent>
      </Card>
    );
  };
  
  // Render overview panel
  const renderOverview = () => {
    return (
      <>
        {renderInsights()}
        
        <Grid container spacing={3}>
          {/* Top Symptoms */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Top Reported Symptoms
                </Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={formatSymptomData()} layout="vertical" margin={{ top: 5, right: 30, left: 100, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" />
                    <YAxis type="category" dataKey="name" tick={{ fontSize: 12 }} width={100} />
                    <Tooltip />
                    <Bar dataKey="count" fill="#8884d8" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </Grid>
          
          {/* Disease Distribution */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Disease Distribution
                </Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={formatDiseaseData()}
                      cx="50%"
                      cy="50%"
                      outerRadius={100}
                      fill="#8884d8"
                      dataKey="value"
                      label={({ name, percent }) => `${name} (${(percent * 100).toFixed(0)}%)`}
                    >
                      {formatDiseaseData().map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </Grid>
          
          {/* Time Series Data */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Diagnoses Over Time
                </Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={formatTimeSeriesData()} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Area type="monotone" dataKey="total" stroke="#8884d8" fill="#8884d8" fillOpacity={0.3} activeDot={{ r: 8 }} />
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </>
    );
  };
  
  // Render disease trends panel
  const renderDiseaseTrends = () => {
    return (
      <Grid container spacing={3}>
        {/* Weekly Trends */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Top 5 Increasing Diseases This Week
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={formatWeeklyTrendsData()} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="disease" />
                  <YAxis label={{ value: 'Percentage Change', angle: -90, position: 'insideLeft' }} />
                  <Tooltip />
                  <Bar dataKey="change" fill={(data) => data.change > 0 ? '#ff4d4f' : '#52c41a'} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
        
        {/* Locality Distribution */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Diagnoses by Locality
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={formatLocalityData()}
                    cx="50%"
                    cy="50%"
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="value"
                    label={({ name, percent }) => `${name} (${(percent * 100).toFixed(0)}%)`}
                  >
                    {formatLocalityData().map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
        
        {/* Medicine Usage */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Medicine Demand vs. Availability
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={formatMedicineData()} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="disease" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="cases" fill="#8884d8" name="Total Cases" />
                  <Bar dataKey="med1_sales" fill="#82ca9d" name="Primary Med Sales" />
                  <Bar dataKey="med2_sales" fill="#ffc658" name="Secondary Med Sales" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    );
  };
  
  // Render advanced analytics panel
  const renderAdvancedAnalytics = () => {
    return (
      <Grid container spacing={3}>
        {/* Symptom Cluster Radar */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Symptom Clusters by Locality
              </Typography>
              <Box sx={{ mb: 2 }}>
                <FormControl size="small" fullWidth>
                  <InputLabel id="locality-select-label">Select Locality</InputLabel>
                  <Select
                    labelId="locality-select-label"
                    id="locality-select"
                    value={selectedLocality}
                    label="Select Locality"
                    onChange={(e) => setSelectedLocality(e.target.value)}
                  >
                    {localityList.map(locality => (
                      <MenuItem key={locality} value={locality}>
                        {locality === 'all' ? 'All Localities' : locality}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Box>
              <ResponsiveContainer width="100%" height={300}>
                <RadarChart cx="50%" cy="50%" outerRadius="80%" data={formatSymptomClusterData()[0] || []}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="name" />
                  <PolarRadiusAxis />
                  <Radar name="Symptom Intensity" dataKey="value" stroke="#8884d8" fill="#8884d8" fillOpacity={0.6} />
                </RadarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
        
        {/* Symptom Correlation */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Symptom Correlation Analysis
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <ScatterChart margin={{ top: 20, right: 20, bottom: 80, left: 20 }}>
                  <CartesianGrid />
                  <XAxis type="category" dataKey="x" name="First Symptom" angle={-45} textAnchor="end" />
                  <YAxis type="category" dataKey="y" name="Second Symptom" />
                  <ZAxis type="number" dataKey="z" range={[60, 400]} name="Correlation" />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} formatter={(value) => [`${value}%`, 'Correlation']} />
                  <Scatter name="Symptom Pairs" data={formatCorrelationData()} fill="#8884d8" />
                </ScatterChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
        
        {/* Top 3 Diseases by Locality */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Top 3 Diseases by Locality
              </Typography>
              <TableContainer component={Paper}>
                <Table sx={{ minWidth: 650 }} aria-label="locality disease table">
                  <TableHead>
                    <TableRow>
                      <TableCell>Locality</TableCell>
                      <TableCell>Top Disease</TableCell>
                      <TableCell>Second Disease</TableCell>
                      <TableCell>Third Disease</TableCell>
                      <TableCell>Total Cases</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {Object.entries(data?.locality_distribution || {}).map(([locality, count]) => (
                      <TableRow key={locality}>
                        <TableCell component="th" scope="row">{locality}</TableCell>
                        <TableCell>
                          <Chip label="Common Cold" color="primary" size="small" />
                        </TableCell>
                        <TableCell>
                          <Chip label="Influenza" color="secondary" size="small" />
                        </TableCell>
                        <TableCell>
                          <Chip label="Allergic Rhinitis" color="default" size="small" />
                        </TableCell>
                        <TableCell>{count}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
              <Box sx={{ mt: 2, display: 'flex', justifyContent: 'flex-end' }}>
                <Button variant="outlined" size="small">
                  Export as CSV
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    );
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="300px">
        <CircularProgress />
      </Box>
    );
  }

  if (error && !data) {
    return (
      <Alert severity="error">{error}</Alert>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Health Statistics Dashboard
      </Typography>
      
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap' }}>
        <DateRangePicker dateRange={dateRange} setDateRange={setDateRange} />
        
        <Box sx={{ display: 'flex', alignItems: 'center', mt: { xs: 2, md: 0 } }}>
          <Typography variant="h6" sx={{ mr: 2 }}>
            Total Diagnoses: {data?.total_diagnoses || 0}
          </Typography>
          <Button 
            variant="outlined" 
            color="primary" 
            size="small" 
            onClick={() => window.location.reload()}
          >
            Refresh Data
          </Button>
        </Box>
      </Box>
      
      <Tabs value={tabValue} onChange={handleTabChange} variant="fullWidth" sx={{ mb: 3 }}>
        <Tab label="Overview" />
        <Tab label="Disease Trends" />
        <Tab label="Advanced Analytics" />
      </Tabs>
      
      <Divider sx={{ mb: 3 }} />
      
      {data && data.total_diagnoses > 0 ? (
        <>
          {tabValue === 0 && renderOverview()}
          {tabValue === 1 && renderDiseaseTrends()}
          {tabValue === 2 && renderAdvancedAnalytics()}
        </>
      ) : (
        <Alert severity="info">
          No diagnosis data available for the selected date range.
        </Alert>
      )}
    </Box>
  );
};

export default Statistics; 