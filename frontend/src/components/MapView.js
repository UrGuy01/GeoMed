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
  Divider,
  Button
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

// Ensure window.google is available before using any of its features
const isGoogleMapsLoaded = () => {
  return typeof window !== 'undefined' && window.google && window.google.maps;
};

// Safely access the google maps value 
const safelyGetGoogleMaps = () => {
  try {
    if (typeof window !== 'undefined' && window.google && window.google.maps) {
      return window.google.maps;
    }
    return null;
  } catch (e) {
    console.error("Error accessing Google Maps:", e);
    return null;
  }
};

const MapView = ({ open, setActive }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [mapData, setMapData] = useState({ points: [], total_points: 0, symptoms: [], top_symptoms: {} });
  const [selectedSymptoms, setSelectedSymptoms] = useState([]);
  const [userLocation, setUserLocation] = useState(null);
  
  // Debug mode related states
  const [debugMode, setDebugMode] = useState(false);
  const [testApiStatus, setTestApiStatus] = useState('');
  const [apiResponse, setApiResponse] = useState(null);
  
  // Mock data for testing when real data isn't available
  const mockMapData = {
    points: [
      { latitude: 13.0827, longitude: 80.2707, symptoms: ["fever"], id: "mock1" }, // Chennai
      { latitude: 12.9716, longitude: 77.5946, symptoms: ["headache"], id: "mock2" }, // Bangalore
      { latitude: 17.3850, longitude: 78.4867, symptoms: ["vomiting"], id: "mock3" }, // Hyderabad
      { latitude: 18.5204, longitude: 73.8567, symptoms: ["vomiting", "fever"], id: "mock4" }, // Pune
      { latitude: 19.0760, longitude: 72.8777, symptoms: ["headache", "fever"], id: "mock5" } // Mumbai
    ],
    total_points: 5,
    symptoms: ["fever", "headache", "vomiting"],
    top_symptoms: { "fever": 3, "headache": 2, "vomiting": 2 }
  };

  // Google Maps API key
  // ... existing code ...

  const [selectedSymptom, setSelectedSymptom] = useState(null);
  const [selectedPoint, setSelectedPoint] = useState(null);
  const [mapRef, setMapRef] = useState(null);
  const [availableSymptoms, setAvailableSymptoms] = useState([]);
  const [isApiLoaded, setIsApiLoaded] = useState(false);
  const [loadScriptError, setLoadScriptError] = useState(null);
  const [useAlternativeDisplay, setUseAlternativeDisplay] = useState(false);
  const [nearbySymptoms, setNearbySymptoms] = useState([]);
  const [isLocating, setIsLocating] = useState(false);
  const [mapsApiLoaded, setMapsApiLoaded] = useState(false);

  // Get the API URL with a fallback
  const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

  // Get user's current location
  const getUserLocation = () => {
    setIsLocating(true);
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const userPos = {
            lat: position.coords.latitude,
            lng: position.coords.longitude
          };
          console.log("User location obtained:", userPos);
          setUserLocation(userPos);
          
          // Fetch data with user location as center
          fetchMapData(userPos);
          setIsLocating(false);
        },
        (error) => {
          console.error("Error getting location:", error);
          setIsLocating(false);
          // Still fetch data without user location
          fetchMapData();
        },
        { enableHighAccuracy: true, timeout: 15000, maximumAge: 10000 }
      );
    } else {
      console.log("Geolocation not supported by this browser");
      setIsLocating(false);
      // Still fetch data without user location
      fetchMapData();
    }
  };

  // Calculate distance between two points in km
  const calculateDistance = (lat1, lon1, lat2, lon2) => {
    const R = 6371; // Radius of the earth in km
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLon = (lon2 - lon1) * Math.PI / 180;
    const a = 
      Math.sin(dLat/2) * Math.sin(dLat/2) +
      Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * 
      Math.sin(dLon/2) * Math.sin(dLon/2); 
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a)); 
    const d = R * c; // Distance in km
    return d;
  };

  // Find symptoms near user's location
  const findNearbySymptoms = (points, userPos, radius = 50) => {
    if (!userPos || !points || points.length === 0) return [];
    
    return points.filter(point => {
      const distance = calculateDistance(
        userPos.lat, userPos.lng, 
        point.latitude, point.longitude
      );
      return distance <= radius; // Within radius kilometers
    });
  };

  // Fetch available symptoms separately
  const fetchAvailableSymptoms = async () => {
    try {
      console.log("Fetching available symptoms from:", `${API_URL}/api/map-data`);
      const response = await axios.get(`${API_URL}/api/map-data`);
      
      // Check if the API response directly includes all_symptoms
      if (response.data && response.data.all_symptoms && response.data.all_symptoms.length > 0) {
        console.log("Available symptoms from API:", response.data.all_symptoms);
        setAvailableSymptoms(response.data.all_symptoms);
        return;
      }
      
      // Fallback: Extract symptoms from points
      if (response.data && response.data.points && response.data.points.length > 0) {
        // Extract unique symptoms from all data points
        const allSymptoms = new Set();
        console.log(`Extracting symptoms from ${response.data.points.length} data points`);
        
        response.data.points.forEach(point => {
          if (Array.isArray(point.symptoms)) {
            point.symptoms.forEach(symptom => {
              if (symptom && typeof symptom === 'string') {
                allSymptoms.add(symptom.trim());
              }
            });
          }
        });
        
        // Convert Set to array and sort
        const symptomArray = Array.from(allSymptoms).sort();
        console.log("Extracted symptoms:", symptomArray);
        setAvailableSymptoms(symptomArray);
      } else {
        console.log("No data points or symptoms available in API response");
      }
    } catch (err) {
      console.error('Error fetching available symptoms:', err);
    }
  };

  // Fetch map data from backend
  const fetchMapData = async (userPos = null) => {
    try {
      setLoading(true);
      
      // Build URL with optional symptom filter
      let url = `${API_URL}/api/map-data`;
      let params = [];
      
      if (selectedSymptom) {
        params.push(`symptom=${encodeURIComponent(selectedSymptom)}`);
      }
      
      // Add user location as center point if available
      if (userPos) {
        params.push(`lat=${userPos.lat}&lng=${userPos.lng}`);
        // Optionally add a radius parameter
        params.push('radius=50'); // 50km radius
      }
      
      if (params.length > 0) {
        url += `?${params.join('&')}`;
      }
      
      console.log("Fetching map data from:", url);
      const response = await axios.get(url);
      setApiResponse(response.data);
      
      // Check if we have valid data points
      if (response.data && response.data.points && response.data.points.length > 0) {
        console.log(`Received ${response.data.points.length} data points from API`);
        setMapData(response.data);
        
        // Find symptoms near user if we have user location
        if (userPos) {
          const nearby = findNearbySymptoms(response.data.points, userPos);
          setNearbySymptoms(nearby);
          console.log(`Found ${nearby.length} symptoms near user's location`);
        }
        
        // Update available symptoms if provided by API
        if (response.data.all_symptoms && response.data.all_symptoms.length > 0) {
          console.log("Setting available symptoms from API response:", response.data.all_symptoms);
          setAvailableSymptoms(response.data.all_symptoms);
        }
        // Otherwise, extract from points as before
        else if (availableSymptoms.length === 0) {
          // Extract unique symptoms from all data points
          const allSymptoms = new Set();
          response.data.points.forEach(point => {
            if (Array.isArray(point.symptoms)) {
              point.symptoms.forEach(symptom => {
                if (symptom && typeof symptom === 'string') {
                  allSymptoms.add(symptom.trim());
                }
              });
            }
          });
          
          // Convert Set to array and sort
          const symptomArray = Array.from(allSymptoms).sort();
          console.log("Setting available symptoms extracted from points:", symptomArray);
          setAvailableSymptoms(symptomArray);
        }
        
        // Fit bounds based on data - only if Google Maps is initialized
        if (mapRef && isGoogleMapsLoaded()) {
          setTimeout(() => {
            console.log("Fitting bounds to points...");
            fitMapToBounds(response.data.points);
          }, 500);
        }
      } else {
        console.log("No map data points returned from API, using mock data for testing");
        if (debugMode) {
          setMapData(mockMapData);
          if (mapRef && isGoogleMapsLoaded()) {
            setTimeout(() => fitMapToBounds(mockMapData.points), 200);
          }
        }
      }
      
      setLoading(false);
    } catch (err) {
      console.error('Error fetching map data:', err);
      setError(`Failed to load location data: ${err.message}`);
      setLoading(false);
      
      if (debugMode) {
        console.log("Using mock data due to API error");
        setMapData(mockMapData);
        setError(null);
        
        // Only try to fit bounds if Maps API is loaded
        if (mapRef && isGoogleMapsLoaded()) {
          setTimeout(() => fitMapToBounds(mockMapData.points), 200);
        }
      }
    }
  };

  // Fit map to show all markers - ONLY call this when Google Maps API is loaded
  const fitMapToBounds = (points) => {
    if (!mapRef || !points || points.length === 0 || !isGoogleMapsLoaded()) {
      console.log("Cannot fit bounds - missing required objects");
      return;
    }
    
    try {
      console.log(`Fitting map to ${points.length} points`);
      
      // Log the points for debugging
      points.forEach((point, index) => {
        console.log(`Point ${index}: lat=${point.latitude}, lng=${point.longitude}`);
      });
      
      const bounds = new window.google.maps.LatLngBounds();
      points.forEach(point => {
        bounds.extend({ lat: point.latitude, lng: point.longitude });
      });
      
      // Add padding to the bounds to make sure markers are visible
      mapRef.fitBounds(bounds, 100); // Add 100 pixels of padding
      
      // If there's only one point, we need to zoom in, as fitBounds won't work well
      if (points.length === 1) {
        mapRef.setCenter({ lat: points[0].latitude, lng: points[0].longitude });
        mapRef.setZoom(10); // Zoom level 10 is good for city-level detail
      }
      
      console.log("Map bounds set successfully");
    } catch (error) {
      console.error("Error fitting bounds:", error);
    }
  };

  // Handle map load and store reference - only called when GoogleMap component loads
  const onMapLoad = (map) => {
    console.log("Map instance loaded successfully");
    setMapRef(map);
    
    // Ensure Google Maps API is fully loaded before attempting to use it
    if (typeof window === 'undefined' || !window.google || !window.google.maps) {
      console.log("Google Maps API not fully loaded yet");
      return;
    }
    
    // Wait until we have both map and data before fitting bounds
    if (isGoogleMapsLoaded() && mapData.points && mapData.points.length > 0) {
      // If we have nearby symptoms, focus on those
      if (userLocation && nearbySymptoms.length > 0) {
        // Use a slightly longer timeout to ensure Maps API is fully initialized
        setTimeout(() => {
          if (isGoogleMapsLoaded()) {
            fitMapToBounds(nearbySymptoms);
          }
        }, 500);
      } else {
        // Otherwise show all symptoms
        setTimeout(() => {
          if (isGoogleMapsLoaded()) {
            fitMapToBounds(mapData.points);
          }
        }, 500);
      }
    }
  };

  useEffect(() => {
    // First, get user location and then fetch data
    getUserLocation();
    
    // Also fetch available symptoms
    fetchAvailableSymptoms();
    
    // Refresh data every 5 minutes
    const interval = setInterval(() => {
      if (userLocation) {
        fetchMapData(userLocation);
      } else {
        fetchMapData();
      }
    }, 300000);
    
    return () => clearInterval(interval);
  }, []);

  // Refetch data when symptom filter changes
  useEffect(() => {
    if (userLocation) {
      fetchMapData(userLocation);
    } else {
      fetchMapData();
    }
  }, [selectedSymptom]);

  // Safe version of getMarkerIcon that won't crash if maps not loaded
  const safeGetMarkerIcon = (point) => {
    try {
      if (!isGoogleMapsLoaded()) return null;
      
      const maps = safelyGetGoogleMaps();
      if (!maps) return null;
      
      // Get the first symptom or a default
      let primarySymptom = (point.symptoms && point.symptoms.length > 0) ? 
        point.symptoms[0] : 'unknown';
        
      // If we're filtering by a specific symptom, use that one
      if (selectedSymptom && point.symptoms.includes(selectedSymptom)) {
        primarySymptom = selectedSymptom;
      }
      
      const color = symptomColors[primarySymptom] || defaultColor;
      
      // Use larger, more visible markers
      return {
        path: maps.SymbolPath.CIRCLE,
        fillColor: color,
        fillOpacity: 0.9,
        strokeWeight: 2,
        strokeColor: '#ffffff',
        scale: 12  // Increased size
      };
    } catch (error) {
      console.error("Error creating marker icon:", error);
      return null;
    }
  };

  // Handle symptom selection
  const handleSymptomChange = (event, newValue) => {
    console.log("Selected symptom:", newValue);
    setSelectedSymptom(newValue);
    
    // Immediately reload with the selected symptom
    if (userLocation) {
      fetchMapData(userLocation);
    } else {
      fetchMapData();
    }
  };

  // Handle user input in the search box
  const handleSymptomInputChange = (event, value) => {
    console.log("Input value:", value);
    // If the user has typed something but we don't have symptoms yet, fetch them
    if (value && availableSymptoms.length === 0) {
      fetchAvailableSymptoms();
    }
  };

  // Handle the Google Maps API loading
  const handleApiLoaded = () => {
    console.log("Google Maps API loaded successfully");
    setIsApiLoaded(true);
    setMapsApiLoaded(true);
  };

  // Handle Google Maps loading errors
  const handleLoadScriptError = (error) => {
    console.error("Google Maps loading error:", error);
    setLoadScriptError("Failed to load Google Maps API. Showing data in alternative format.");
    setUseAlternativeDisplay(true); // Switch to alternative display mode
  };

  // Toggle debug mode
  const toggleDebugMode = () => {
    const newMode = !debugMode;
    setDebugMode(newMode);
    if (newMode && (!mapData.points || mapData.points.length === 0)) {
      setMapData(mockMapData);
    }
  };

  // Test API connection directly
  const testApiConnection = async () => {
    setTestApiStatus('Testing API connection...');
    try {
      // Test the map-data endpoint
      const mapResponse = await axios.get(`${API_URL}/api/map-data`);
      console.log('Map API Response:', mapResponse.data);
      
      if (mapResponse.data && mapResponse.data.points) {
        if (mapResponse.data.points.length > 0) {
          setTestApiStatus(`Success! Received ${mapResponse.data.points.length} data points from map-data API.`);
          // If in debug mode but using mock data, switch to real data
          if (debugMode && mapData === mockMapData) {
            setMapData(mapResponse.data);
          }
        } else {
          setTestApiStatus('API connection successful, but no data points were returned.');
        }
      } else {
        setTestApiStatus('API connection successful, but response format is incorrect.');
      }
      
      setApiResponse(mapResponse.data);
    } catch (error) {
      console.error('API Test Error:', error);
      setTestApiStatus(`API test failed: ${error.message}`);
      
      // Try symptoms endpoint as a fallback
      try {
        const symptomsResponse = await axios.get(`${API_URL}/api/symptoms`);
        if (symptomsResponse.data) {
          setTestApiStatus(`Map-data API failed, but symptoms API works. This indicates an issue with the map-data endpoint.`);
        }
      } catch (fallbackError) {
        setTestApiStatus(`Both APIs failed. Check if the backend server is running at ${API_URL}`);
      }
    }
  };

  // Force refresh markers
  const forceRefreshMarkers = () => {
    console.log("Force refreshing markers...");
    
    if (mapData.points && mapData.points.length > 0) {
      console.log("Current map points:");
      mapData.points.forEach((point, index) => {
        console.log(`Point ${index}: lat=${point.latitude}, lng=${point.longitude}, symptoms=${point.symptoms}`);
      });
    } else {
      console.log("No points available to display");
    }
    
    // Force map to recenter and zoom to markers
    if (mapRef && mapData.points && mapData.points.length > 0) {
      console.log("Forcing map to fit to bounds");
      
      // Set immediate zoom to a wider view first
      mapRef.setZoom(5);
      
      // Then fit to bounds after a short delay
      setTimeout(() => {
        fitMapToBounds(mapData.points);
      }, 200);
    }
    
    // If we have no points but we're in debug mode, use mock data
    if ((!mapData.points || mapData.points.length === 0) && debugMode) {
      console.log("Using mock data for testing");
      setMapData(mockMapData);
      
      if (mapRef) {
        setTimeout(() => {
          fitMapToBounds(mockMapData.points);
        }, 200);
      }
    }
  };

  // Add a useEffect to handle map loading timeout
  useEffect(() => {
    // Set a loading timeout of 10 seconds
    if (!mapsApiLoaded && !useAlternativeDisplay) {
      const loadingTimeout = setTimeout(() => {
        console.log("Google Maps loading timed out after 10 seconds");
        setUseAlternativeDisplay(true);
        setLoadScriptError("Google Maps loading timed out. Showing data in alternative format.");
      }, 10000);
      
      return () => clearTimeout(loadingTimeout);
    }
  }, [mapsApiLoaded, useAlternativeDisplay]);

  // Render loading state
  if (loading && mapData.points.length === 0) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  // Render the symptom color legend
  const renderSymptomLegend = () => {
    // Get the active symptoms based on current data
    const activeSymptoms = new Set();
    
    mapData.points.forEach(point => {
      if (Array.isArray(point.symptoms)) {
        if (selectedSymptom && point.symptoms.includes(selectedSymptom)) {
          activeSymptoms.add(selectedSymptom);
        } else {
          point.symptoms.forEach(s => activeSymptoms.add(s));
        }
      }
    });
    
    return (
      <Box sx={{ mb: 2 }}>
        <Typography variant="h6" gutterBottom>Symptom Legend</Typography>
        <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap sx={{ gap: 1 }}>
          {Array.from(activeSymptoms).map(symptom => (
            <Chip
              key={symptom}
              label={symptom.replace('_', ' ')}
              sx={{
                bgcolor: symptomColors[symptom] || defaultColor,
                color: '#fff',
                fontWeight: 'bold',
                '&:hover': { opacity: 0.9 }
              }}
              onClick={() => setSelectedSymptom(symptom === selectedSymptom ? null : symptom)}
              variant={symptom === selectedSymptom ? "outlined" : "filled"}
            />
          ))}
        </Stack>
      </Box>
    );
  };

  // Alternative display for map data when Google Maps fails
  const renderAlternativeMapDisplay = () => {
    return (
      <Box sx={{ mt: 2, p: 2, border: '1px solid #e0e0e0', borderRadius: 1, bgcolor: '#f9f9f9' }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">Map Data (List View)</Typography>
          {loadScriptError ? (
            <Typography variant="caption" color="error">{loadScriptError}</Typography>
          ) : (
            <Button 
              variant="contained" 
              size="small" 
              onClick={() => {
                setUseAlternativeDisplay(false);
                setLoadScriptError(null);
              }}
            >
              Try Map View Again
            </Button>
          )}
        </Box>
        
        {mapData.points && mapData.points.length > 0 ? (
          <>
            <Typography variant="body2" gutterBottom>
              Showing {mapData.points.length} data points:
            </Typography>
            <Box sx={{ maxHeight: '400px', overflow: 'auto', mt: 1 }}>
              {mapData.points.map((point, index) => (
                <Card key={point.id || index} variant="outlined" sx={{ mb: 1, p: 1 }}>
                  <CardContent sx={{ p: 1, '&:last-child': { pb: 1 } }}>
                    <Typography variant="subtitle2" sx={{ fontWeight: 'bold' }}>
                      {point.city || `Location ${index + 1}`} 
                      {point.country ? `, ${point.country}` : ''}
                    </Typography>
                    <Typography variant="body2">
                      Coordinates: {point.latitude.toFixed(4)}, {point.longitude.toFixed(4)}
                    </Typography>
                    <Box sx={{ 
                      display: 'flex', 
                      flexWrap: 'wrap', 
                      gap: '4px', 
                      mt: 1 
                    }}>
                      {Array.isArray(point.symptoms) && point.symptoms.map(symptom => (
                        <Chip
                          key={symptom}
                          label={symptom.replace('_', ' ')}
                          size="small"
                          sx={{
                            bgcolor: symptomColors[symptom] || defaultColor,
                            color: '#fff',
                            fontSize: '0.7rem'
                          }}
                        />
                      ))}
                    </Box>
                    {point.diagnosis && (
                      <Typography variant="body2" sx={{ mt: 1 }}>
                        Diagnosis: {point.diagnosis}
                      </Typography>
                    )}
                  </CardContent>
                </Card>
              ))}
            </Box>
          </>
        ) : (
          <Typography color="textSecondary">No data points available.</Typography>
        )}
      </Box>
    );
  };

  // Render the debug panel component
  const renderDebugPanel = () => {
    return (
      <Box sx={{ mt: 2, p: 2, border: '1px solid #e0e0e0', borderRadius: 1, bgcolor: '#f5f5f5' }}>
        <Typography variant="h6" gutterBottom sx={{ display: 'flex', justifyContent: 'space-between' }}>
          Debug Panel
          <Chip 
            label={debugMode ? "Using Mock Data" : "Using Real API"} 
            color={debugMode ? "secondary" : "primary"} 
            size="small"
            onClick={toggleDebugMode}
          />
        </Typography>
        
        <Grid container spacing={2} sx={{ mb: 1 }}>
          <Grid item xs={12} md={4}>
            <Typography variant="body2">API URL: {API_URL}/api/map-data</Typography>
            <Typography variant="body2">
              Data Status: {loading ? 'Loading...' : (mapData.points && mapData.points.length > 0 ? 'Data Loaded' : 'No Data')}
            </Typography>
            {testApiStatus && (
              <Typography variant="body2" color="primary" sx={{ mt: 1, fontWeight: 'medium' }}>
                {testApiStatus}
              </Typography>
            )}
          </Grid>
          <Grid item xs={12} md={4}>
            <Button 
              variant="outlined" 
              size="small" 
              onClick={testApiConnection}
              fullWidth
            >
              Test API Connection
            </Button>
          </Grid>
          <Grid item xs={12} md={4}>
            <Button 
              variant="outlined" 
              color="secondary"
              size="small" 
              onClick={forceRefreshMarkers}
              fullWidth
            >
              Force Refresh Markers
            </Button>
          </Grid>
        </Grid>
        
        <Box sx={{ mt: 1 }}>
          <Typography variant="body2">API Response:</Typography>
          <Box 
            component="pre" 
            sx={{ 
              mt: 1, 
              p: 1, 
              bgcolor: '#f0f0f0', 
              border: '1px solid #ddd', 
              borderRadius: 1,
              fontSize: '0.75rem',
              maxHeight: '200px',
              overflow: 'auto'
            }}
          >
            {apiResponse ? JSON.stringify(apiResponse, null, 2) : 'No response data'}
          </Box>
        </Box>
      </Box>
    );
  };

  // Render markers with clustering
  const renderMarkers = (clusterer) => {
    if (!mapData.points || mapData.points.length === 0) {
      console.log("No points to render markers for");
      return null;
    }
    
    console.log(`Rendering ${mapData.points.length} markers`);
    
    // Create markers with fixed z-index to ensure visibility
    return mapData.points.map((point, index) => {
      console.log(`Creating marker for point ${index}: lat=${point.latitude}, lng=${point.longitude}`);
      return (
        <Marker
          key={point.id || `marker-${index}-${point.latitude}-${point.longitude}`}
          position={{ lat: point.latitude, lng: point.longitude }}
          icon={safeGetMarkerIcon(point)}
          clusterer={clusterer}
          onClick={() => setSelectedPoint(point)}
          zIndex={10} // Ensure markers are on top
          animation={isGoogleMapsLoaded() && window.google.maps.Animation ? window.google.maps.Animation.DROP : null} // Add animation for visibility
        />
      );
    });
  };

  // Improved renderMap function with better error handling
  const renderMap = () => {
    return (
      <LoadScript
        googleMapsApiKey={process.env.REACT_APP_GOOGLE_MAPS_API_KEY}
        loadingElement={
          <Box display="flex" justifyContent="center" alignItems="center" height="600px">
            <CircularProgress />
            <Typography variant="body2" sx={{ ml: 2 }}>Loading Google Maps... Please wait</Typography>
          </Box>
        }
        onLoad={handleApiLoaded}
        onError={handleLoadScriptError}
        id="google-map-script"
        preventGoogleFontsLoading={true}
        language="en"
        libraries={["places", "geometry", "visualization"]}
      >
        {mapsApiLoaded && isGoogleMapsLoaded() ? (
          <GoogleMap
            mapContainerStyle={containerStyle}
            center={userLocation || center}
            zoom={3}
            onLoad={onMapLoad}
            options={{
              fullscreenControl: false,
              streetViewControl: false,
              mapTypeControl: true,
              zoomControl: true
            }}
          >
            {/* Show user's location marker if available */}
            {userLocation && safelyGetGoogleMaps() && (
              <Marker
                position={userLocation}
                icon={{
                  path: safelyGetGoogleMaps().SymbolPath.CIRCLE,
                  fillColor: '#4285F4',
                  fillOpacity: 1,
                  strokeWeight: 2,
                  strokeColor: '#ffffff',
                  scale: 10
                }}
                title="Your Location"
                zIndex={5}
              />
            )}
          
            {/* Render markers with clustering */}
            {mapData.points && mapData.points.length > 0 && (
              <MarkerClusterer>
                {(clusterer) => renderMarkers(clusterer)}
              </MarkerClusterer>
            )}
            
            {/* Info window for selected marker */}
            {selectedPoint && (
              <InfoWindow
                position={{ lat: selectedPoint.latitude, lng: selectedPoint.longitude }}
                onCloseClick={() => setSelectedPoint(null)}
              >
                <div style={{ padding: '8px', maxWidth: '300px' }}>
                  <Typography variant="subtitle1" sx={{ fontWeight: 'bold', mb: 1 }}>
                    Reported Symptoms
                  </Typography>
                  <Typography variant="body2">
                    {Array.isArray(selectedPoint.symptoms) 
                      ? selectedPoint.symptoms.map(s => s.replace('_', ' ')).join(', ')
                      : 'No symptoms reported'}
                  </Typography>
                  <Divider sx={{ my: 1 }} />
                  {selectedPoint.diagnosis && (
                    <Typography variant="body2">
                      Diagnosis: {selectedPoint.diagnosis}
                    </Typography>
                  )}
                  {selectedPoint.confidence !== undefined && (
                    <Typography variant="body2">
                      Confidence: {(selectedPoint.confidence || 0).toFixed(1)}%
                    </Typography>
                  )}
                  {selectedPoint.city && (
                    <Typography variant="body2">
                      Location: {selectedPoint.city}{selectedPoint.country ? `, ${selectedPoint.country}` : ''}
                    </Typography>
                  )}
                  {selectedPoint.created_at && (
                    <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                      Reported: {new Date(selectedPoint.created_at).toLocaleString()}
                    </Typography>
                  )}
                </div>
              </InfoWindow>
            )}
          </GoogleMap>
        ) : (
          <Box display="flex" justifyContent="center" alignItems="center" height="600px">
            <CircularProgress />
            <Typography variant="body2" sx={{ ml: 2 }}>Loading Google Maps...</Typography>
          </Box>
        )}
      </LoadScript>
    );
  };

  return (
    <Box p={3}>
      <Paper elevation={3} sx={{ p: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h4">
            Symptom Map
          </Typography>
          <Box>
            <Button 
              variant="outlined" 
              size="small" 
              onClick={getUserLocation}
              disabled={isLocating}
              sx={{ mr: 1 }}
              startIcon={isLocating ? <CircularProgress size={16} /> : null}
            >
              {isLocating ? 'Getting Location...' : 'Find Nearby Symptoms'}
            </Button>
            {!useAlternativeDisplay && (
              <Button
                variant="outlined"
                size="small"
                onClick={() => setUseAlternativeDisplay(true)}
                sx={{ mr: 1 }}
              >
                Show List View
              </Button>
            )}
            <Chip 
              label={debugMode ? "Debug Mode ON" : "Debug Mode"}
              color={debugMode ? "secondary" : "default"}
              onClick={toggleDebugMode}
              size="small"
            />
          </Box>
        </Box>
        
        <Grid container spacing={2} sx={{ mb: 3 }}>
          {/* Search Filter */}
          <Grid item xs={12} md={8}>
            <Autocomplete
              options={availableSymptoms || []}
              value={selectedSymptom}
              onChange={handleSymptomChange}
              onInputChange={handleSymptomInputChange}
              renderInput={(params) => (
                <TextField 
                  {...params} 
                  label="Search for a symptom" 
                  fullWidth 
                  placeholder="Start typing or select from available symptoms"
                  helperText={availableSymptoms.length === 0 ? "Loading symptoms..." : ""}
                />
              )}
              renderOption={(props, option) => {
                const { key, ...otherProps } = props;
                return (
                  <li {...otherProps} key={key}>
                    <Box component="span" sx={{ 
                      display: 'inline-block', 
                      width: '12px', 
                      height: '12px', 
                      borderRadius: '50%', 
                      bgcolor: symptomColors[option] || defaultColor,
                      mr: 1
                    }}/>
                    {option.replace('_', ' ')}
                  </li>
                );
              }}
              getOptionLabel={(option) => option ? option.replace('_', ' ') : ''}
              isOptionEqualToValue={(option, value) => option === value}
              filterOptions={(options, state) => {
                const inputValue = state.inputValue.toLowerCase();
                return options.filter(option => 
                  option.toLowerCase().includes(inputValue)
                );
              }}
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
                {selectedSymptom && mapData.points && (
                  <Typography variant="body2">
                    Matching points: {mapData.points.filter(p => 
                      Array.isArray(p.symptoms) && p.symptoms.includes(selectedSymptom)
                    ).length}
                  </Typography>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
        
        {/* Common Symptom Color Legend */}
        {mapData.points && mapData.points.length > 0 && renderSymptomLegend()}
        
        {error ? (
          <Box p={3}>
            <Typography color="error">{error}</Typography>
          </Box>
        ) : useAlternativeDisplay ? (
          renderAlternativeMapDisplay()
        ) : loadScriptError ? (
          <Box>
            <Typography color="error">{loadScriptError}</Typography>
            {renderAlternativeMapDisplay()}
          </Box>
        ) : (
          renderMap()
        )}
        
        {/* Nearby Symptoms Stat Card */}
        {nearbySymptoms.length > 0 && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="h6" gutterBottom>
              Nearby Symptoms ({nearbySymptoms.length})
            </Typography>
            <Typography variant="body2">
              Top symptoms in your area:
            </Typography>
            <Box sx={{ mt: 1 }}>
              {/* Group nearby symptoms and show counts */}
              {(() => {
                const symptomCounts = {};
                nearbySymptoms.forEach(point => {
                  if (Array.isArray(point.symptoms)) {
                    point.symptoms.forEach(s => {
                      symptomCounts[s] = (symptomCounts[s] || 0) + 1;
                    });
                  }
                });
                
                return Object.entries(symptomCounts)
                  .sort((a, b) => b[1] - a[1])
                  .slice(0, 5)
                  .map(([symptom, count]) => (
                    <Chip
                      key={symptom}
                      label={`${symptom.replace('_', ' ')} (${count})`}
                      sx={{
                        m: 0.5,
                        bgcolor: symptomColors[symptom] || defaultColor,
                        color: '#fff',
                        fontWeight: 'bold'
                      }}
                      onClick={() => setSelectedSymptom(symptom === selectedSymptom ? null : symptom)}
                      variant={symptom === selectedSymptom ? "outlined" : "filled"}
                    />
                  ));
              })()}
            </Box>
          </Box>
        )}
        
        {/* Debug Panel - only shown in debug mode */}
        {debugMode && renderDebugPanel()}
      </Paper>
    </Box>
  );
};

export default MapView; 