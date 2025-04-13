import React, { useEffect, useState } from 'react';
import { 
  GoogleMap, 
  LoadScript, 
  Marker, 
  InfoWindow, 
  MarkerClusterer 
} from '@react-google-maps/api';
import { 
  Box, 
  Typography, 
  Paper, 
  CircularProgress, 
  TextField,
  Grid,
  Chip,
  Stack,
  Card,
  CardContent,
  Autocomplete,
  Divider
} from '@mui/material';
import axios from 'axios';

const containerStyle = {
  width: '100%',
  height: '600px'
};

// Default center on global view
const center = {
  lat: 20.5937,
  lng: 78.9629
};

// Generate colors for different common symptoms
const symptomColors = {
  'itching': '#FF5733',
  'skin_rash': '#33FF57',
  'nodal_skin_eruptions': '#3357FF',
  'continuous_sneezing': '#FF33A8',
  'shivering': '#FFFF33',
  'chills': '#33FFFF',
  'joint_pain': '#FF33FF',
  'stomach_pain': '#FFA833',
  'acidity': '#A833FF',
  'vomiting': '#FF3333',
  'fatigue': '#33FFAA',
  'weight_loss': '#AA33FF',
  'cough': '#FFAA33',
  'high_fever': '#33AAFF',
  'headache': '#FF33AA'
};

// Default color for symptoms not in the list
const defaultColor = '#808080';

const MapView = () => {
  const [mapData, setMapData] = useState({
    points: [],
    total_points: 0,
    top_symptoms: []
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedSymptom, setSelectedSymptom] = useState('');
  const [selectedPoint, setSelectedPoint] = useState(null);
  const [mapRef, setMapRef] = useState(null);
  const [availableSymptoms, setAvailableSymptoms] = useState([]);

  // Get the API URL with a fallback
  const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

  // Fetch map data from backend
  const fetchMapData = async () => {
    try {
      setLoading(true);
      
      // Build URL with optional symptom filter
      let url = `${API_URL}/api/map-data`;
      if (selectedSymptom) {
        url += `?symptom=${selectedSymptom}`;
      }
      
      console.log("Fetching map data from:", url);
      const response = await axios.get(url);
      setMapData(response.data);
      
      // Extract unique symptoms from all data points
      const allSymptoms = new Set();
      response.data.points.forEach(point => {
        if (Array.isArray(point.symptoms)) {
          point.symptoms.forEach(symptom => allSymptoms.add(symptom));
        }
      });
      
      // Update available symptoms for search
      if (availableSymptoms.length === 0) {
        setAvailableSymptoms(Array.from(allSymptoms));
      }
      
      // If we have points, adjust map to show them all
      if (response.data.points && response.data.points.length > 0 && mapRef) {
        fitMapToBounds(response.data.points);
      }
      
      setLoading(false);
    } catch (err) {
      console.error('Error fetching map data:', err);
      setError('Failed to load location data');
      setLoading(false);
    }
  };

  // Fit map to show all markers
  const fitMapToBounds = (points) => {
    if (!mapRef || points.length === 0) return;
    
    const bounds = new window.google.maps.LatLngBounds();
    points.forEach(point => {
      bounds.extend({ lat: point.latitude, lng: point.longitude });
    });
    
    mapRef.fitBounds(bounds);
  };

  useEffect(() => {
    // First, fetch all available symptoms
    const fetchSymptoms = async () => {
      try {
        console.log("Fetching symptoms from:", `${API_URL}/api/symptoms`);
        const response = await axios.get(`${API_URL}/api/symptoms`);
        if (response.data && response.data.symptoms) {
          setAvailableSymptoms(response.data.symptoms);
        }
      } catch (err) {
        console.error('Error fetching symptoms:', err);
      }
    };
    
    fetchSymptoms();
    fetchMapData();
    
    // Refresh data every 5 minutes
    const interval = setInterval(fetchMapData, 300000);
    return () => clearInterval(interval);
  }, []);

  // Refetch data when symptom filter changes
  useEffect(() => {
    fetchMapData();
  }, [selectedSymptom]);

  // Handle map load and store reference
  const onMapLoad = (map) => {
    setMapRef(map);
    if (mapData.points && mapData.points.length > 0) {
      fitMapToBounds(mapData.points);
    }
  };

  // Get marker icon based on symptom
  const getMarkerIcon = (point) => {
    // Get the first symptom or a default
    let primarySymptom = (point.symptoms && point.symptoms.length > 0) ? 
      point.symptoms[0] : 'unknown';
      
    // If we're filtering by a specific symptom, use that one
    if (selectedSymptom && point.symptoms.includes(selectedSymptom)) {
      primarySymptom = selectedSymptom;
    }
    
    const color = symptomColors[primarySymptom] || defaultColor;
    
    return {
      path: window.google.maps.SymbolPath.CIRCLE,
      fillColor: color,
      fillOpacity: 0.8,
      strokeWeight: 1,
      strokeColor: '#ffffff',
      scale: 8
    };
  };

  // Handle symptom selection
  const handleSymptomChange = (event, newValue) => {
    setSelectedSymptom(newValue || '');
  };

  if (loading && mapData.points.length === 0) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box p={3}>
      <Paper elevation={3} sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom>
          Symptom Map
        </Typography>
        
        <Grid container spacing={2} sx={{ mb: 3 }}>
          {/* Search Filter */}
          <Grid item xs={12} md={8}>
            <Autocomplete
              options={availableSymptoms}
              value={selectedSymptom}
              onChange={handleSymptomChange}
              renderInput={(params) => (
                <TextField 
                  {...params} 
                  label="Search for a symptom" 
                  fullWidth 
                  placeholder="Leave empty to see all symptoms"
                />
              )}
              renderOption={(props, option) => (
                <li {...props}>
                  {option.replace('_', ' ')}
                </li>
              )}
              getOptionLabel={(option) => option.replace('_', ' ')}
            />
          </Grid>
          
          {/* Stats */}
          <Grid item xs={12} md={4}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="h6" gutterBottom>Map Statistics</Typography>
                <Typography variant="body2">
                  Total Data Points: {mapData.total_points || 0}
                </Typography>
                {selectedSymptom && (
                  <Typography variant="body2">
                    Filtering by: {selectedSymptom.replace('_', ' ')}
                  </Typography>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
        
        {/* Common Symptom Color Legend */}
        <Box sx={{ mb: 2 }}>
          <Typography variant="h6" gutterBottom>Symptom Legend</Typography>
          <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
            {Object.entries(symptomColors).map(([symptom, color]) => (
              <Chip
                key={symptom}
                label={symptom.replace('_', ' ')}
                sx={{ 
                  backgroundColor: color, 
                  color: '#fff',
                  mb: 1,
                  textShadow: '0px 0px 2px rgba(0,0,0,0.7)'
                }}
                onClick={() => setSelectedSymptom(symptom)}
              />
            ))}
          </Stack>
        </Box>
        
        <Divider sx={{ my: 2 }} />
        
        {/* Top Symptoms */}
        {mapData.top_symptoms && mapData.top_symptoms.length > 0 && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="h6" gutterBottom>Top Symptoms</Typography>
            <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
              {mapData.top_symptoms.map((symptom) => (
                <Chip
                  key={symptom.name}
                  label={`${symptom.name.replace('_', ' ')} (${symptom.count})`}
                  variant="outlined"
                  sx={{ mb: 1 }}
                  onClick={() => setSelectedSymptom(symptom.name)}
                />
              ))}
            </Stack>
          </Box>
        )}
        
        {error ? (
          <Box p={3}>
            <Typography color="error">{error}</Typography>
          </Box>
        ) : (
          <LoadScript
            googleMapsApiKey={process.env.REACT_APP_GOOGLE_MAPS_API_KEY}
            libraries={['visualization']}
          >
            <GoogleMap
              mapContainerStyle={containerStyle}
              center={center}
              zoom={3}
              onLoad={onMapLoad}
            >
              {/* Render markers with clustering */}
              {mapData.points && mapData.points.length > 0 && (
                <MarkerClusterer>
                  {(clusterer) => (
                    <div>
                      {mapData.points.map((point) => (
                        <Marker
                          key={point.id}
                          position={{ lat: point.latitude, lng: point.longitude }}
                          icon={getMarkerIcon(point)}
                          clusterer={clusterer}
                          onClick={() => setSelectedPoint(point)}
                        />
                      ))}
                    </div>
                  )}
                </MarkerClusterer>
              )}
              
              {/* Info window for selected marker */}
              {selectedPoint && (
                <InfoWindow
                  position={{ lat: selectedPoint.latitude, lng: selectedPoint.longitude }}
                  onCloseClick={() => setSelectedPoint(null)}
                >
                  <div style={{ padding: '5px', maxWidth: '300px' }}>
                    <Typography variant="h6">Reported Symptoms</Typography>
                    <Typography variant="body2">
                      {selectedPoint.symptoms.map(s => s.replace('_', ' ')).join(', ')}
                    </Typography>
                    <Divider sx={{ my: 1 }} />
                    <Typography variant="body2">Diagnosis: {selectedPoint.diagnosis}</Typography>
                    <Typography variant="body2">Confidence: {selectedPoint.confidence.toFixed(1)}%</Typography>
                    {selectedPoint.city && (
                      <Typography variant="body2">Location: {selectedPoint.city}, {selectedPoint.country}</Typography>
                    )}
                    <Typography variant="caption">Reported: {new Date(selectedPoint.created_at).toLocaleString()}</Typography>
                  </div>
                </InfoWindow>
              )}
            </GoogleMap>
          </LoadScript>
        )}
      </Paper>
    </Box>
  );
};

export default MapView; 