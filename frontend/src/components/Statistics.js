import React, { useState, useEffect } from 'react';
import { Typography, Grid, Card, CardContent, Box, CircularProgress, Alert } from '@mui/material';
import { BarChart, Bar, PieChart, Pie, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';
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
            end_date: endDateStr
          }
        });
        
        setData(response.data);
        setError(null);
      } catch (err) {
        console.error('Error fetching statistics:', err);
        setError('Failed to load statistics. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [dateRange]);

  // Format symptom data for bar chart
  const formatSymptomData = () => {
    if (!data || !data.top_symptoms) return [];
    
    return data.top_symptoms.map(([symptom, count]) => ({
      name: symptom,
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

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="300px">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error">{error}</Alert>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Health Statistics Dashboard
      </Typography>
      
      <DateRangePicker dateRange={dateRange} setDateRange={setDateRange} />
      
      {data && data.total_diagnoses > 0 ? (
        <>
          <Box mb={3}>
            <Typography variant="h6" gutterBottom>
              Total Diagnoses: {data.total_diagnoses}
            </Typography>
          </Box>
          
          <Grid container spacing={3}>
            {/* Symptom Distribution */}
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Top Symptoms
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
                    <LineChart data={formatTimeSeriesData()} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="total" stroke="#8884d8" activeDot={{ r: 8 }} />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
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