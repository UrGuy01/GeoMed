import React, { useEffect, useState } from 'react';
import { GoogleMap, LoadScript, HeatmapLayer } from '@react-google-maps/api';
import { Box, Typography, Paper, CircularProgress } from '@mui/material';
import axios from 'axios';

const containerStyle = {
  width: '100%',
  height: '600px'
};

const center = {
  lat: 20.5937,
  lng: 78.9629
};

const MapView = () => {
  const [locations, setLocations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get('http://localhost:5000/api/statistics');
        setLocations(response.data.locations);
        setLoading(false);
      } catch (err) {
        setError('Failed to load location data');
        setLoading(false);
      }
    };

    fetchData();
    // Refresh data every 5 minutes
    const interval = setInterval(fetchData, 300000);
    return () => clearInterval(interval);
  }, []);

  const getHeatmapData = () => {
    return locations.map(location => ({
      location: new window.google.maps.LatLng(location.lat, location.lng),
      weight: location.symptoms.length
    }));
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box p={3}>
        <Paper elevation={3} sx={{ p: 3 }}>
          <Typography color="error">{error}</Typography>
        </Paper>
      </Box>
    );
  }

  return (
    <Box p={3}>
      <Paper elevation={3} sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom>
          Symptom Distribution Map
        </Typography>
        <Typography variant="body1" paragraph>
          This heatmap shows the concentration of reported symptoms across different regions.
          Darker areas indicate higher symptom frequency.
        </Typography>
        <LoadScript
          googleMapsApiKey={process.env.REACT_APP_GOOGLE_MAPS_API_KEY}
          libraries={['visualization']}
        >
          <GoogleMap
            mapContainerStyle={containerStyle}
            center={center}
            zoom={5}
          >
            <HeatmapLayer
              data={getHeatmapData()}
              options={{
                radius: 20,
                opacity: 0.6
              }}
            />
          </GoogleMap>
        </LoadScript>
      </Paper>
    </Box>
  );
};

export default MapView; 