import React, { useEffect, useState, useMemo } from 'react';
import { 
  GoogleMap, 
  useJsApiLoader,
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
  height: '600px',
  border: '1px solid #ccc',
  borderRadius: '4px',
  position: 'relative',
  overflow: 'hidden'
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

// Safely check if KML functionality is available
const isKmlAvailable = () => {
  try {
    return typeof window !== 'undefined' && 
           window.google && 
           window.google.maps && 
           window.google.maps.KmlLayer;
  } catch (e) {
    console.error("Error checking KML availability:", e);
    return false;
  }
};

// Add a helper method to safely check for visualization library
const isVisualizationAvailable = () => {
  try {
    // Don't actually access the visualization property, just check if it exists
    return !!(typeof window !== 'undefined' && 
      window.google && 
      window.google.maps);
  } catch (e) {
    console.error("Error checking visualization availability:", e);
    return false;
  }
};

// Helper function to check if markers can be properly created
const canCreateMarkers = () => {
  return typeof window !== 'undefined' && 
         window.google && 
         window.google.maps && 
         window.google.maps.Marker;
};

const MapView = ({ open, setActive }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [mapData, setMapData] = useState({ points: [], total_points: 0, symptoms: [], top_symptoms: {} });
  const [selectedSymptoms, setSelectedSymptoms] = useState([]);
  const [userLocation, setUserLocation] = useState(null);
  
  // Create stable reference to libraries to prevent reloading
  const mapLibraries = useMemo(() => ["places"], []);
  
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
  const [useAlternativeDisplay, setUseAlternativeDisplay] = useState(false);
  const [nearbySymptoms, setNearbySymptoms] = useState([]);

  // Replace LoadScript with useJsApiLoader
  const { isLoaded, loadError } = useJsApiLoader({
    id: 'google-map-script',
    googleMapsApiKey: process.env.REACT_APP_GOOGLE_MAPS_API_KEY,
    libraries: ['places', 'visualization', 'geometry']
  });

  // No need for mapsApiLoaded state anymore, use isLoaded from useJsApiLoader
  const [isApiLoaded, setIsApiLoaded] = useState(false);
  const [loadScriptError, setLoadScriptError] = useState(null);
  const [isLocating, setIsLocating] = useState(false);

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
      
      try {
        const response = await axios.get(url);
        setApiResponse(response.data);
        
        // Check if we have valid data points
        if (response.data && response.data.points && response.data.points.length > 0) {
          console.log(`Received ${response.data.points.length} data points from API`);
          
          // Examine the data format to debug marker issues
          response.data.points.forEach((point, index) => {
            console.log(`Point ${index}: lat=${point.latitude}, lng=${point.longitude}, symptoms=${point.symptoms}`);
            // Check for potential issues
            if (!point.latitude || !point.longitude) {
              console.warn(`  Warning: Point ${index} missing coordinates`);
            } else if (typeof point.latitude !== 'number' || typeof point.longitude !== 'number') {
              console.warn(`  Warning: Point ${index} has non-numeric coordinates`);
            }
          });
          
          setMapData(response.data);
          setError(null);
          
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
        } else {
          console.log("No data points returned from API, using mock data");
          setMapData(mockMapData);
        }
      } catch (apiError) {
        console.error("API request failed:", apiError);
        setError(`Failed to load location data. ${apiError.message}`);
        
        // Use mock data when API fails
        console.log("Using mock data due to API error");
        setMapData(mockMapData);
      }
      
      setLoading(false);
    } catch (err) {
      console.error('Error in fetchMapData:', err);
      setError(`An error occurred: ${err.message}`);
      setLoading(false);
      
      // Always use mock data when there's an error
      console.log("Using mock data due to error");
      setMapData(mockMapData);
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
      
      // Create a simple bounds literal object directly instead of using the API
      let north = -90, south = 90, east = -180, west = 180;
      
      // Find min/max coordinates
      let validPointsCount = 0;
      points.forEach(point => {
        if (point && 
            typeof point.latitude === 'number' && 
            typeof point.longitude === 'number' &&
            !isNaN(point.latitude) && 
            !isNaN(point.longitude)) {
          
          const lat = parseFloat(point.latitude);
          const lng = parseFloat(point.longitude);
          
          north = Math.max(north, lat);
          south = Math.min(south, lat);
          east = Math.max(east, lng);
          west = Math.min(west, lng);
          
          validPointsCount++;
        }
      });
      
      if (validPointsCount === 0) {
        console.error("No valid points to fit bounds");
        return;
      }
      
      // Add some padding
      const padding = 0.1; // degrees
      north += padding;
      south -= padding;
      east += padding;
      west -= padding;
      
      // Create bounds literal object
      const bounds = {
        north: north,
        south: south,
        east: east,
        west: west
      };
      
      console.log("Created bounds:", bounds);
      
      // Apply the bounds to the map
      console.log("Applying bounds to map");
      mapRef.fitBounds(bounds);
      
      // If there's only one point, we need to zoom in
      if (validPointsCount === 1) {
        mapRef.setCenter({ 
          lat: parseFloat(points[0].latitude), 
          lng: parseFloat(points[0].longitude) 
        });
        mapRef.setZoom(10); // Zoom level 10 is good for city-level detail
      }
      
      console.log("Map bounds set successfully");
    } catch (error) {
      console.error("Error fitting bounds:", error);
    }
  };

  // Clean up handleApiLoaded since we're using useJsApiLoader now
  const handleApiLoaded = () => {
    console.log("Google Maps API has loaded successfully");
    setIsApiLoaded(true);
  };

  // Update onMapLoad to be more defensive
  const onMapLoad = (map) => {
    console.log("Map loaded successfully");
    
    if (!map) {
      console.error("Map reference is null");
      return;
    }
    
    // Store map reference for later use
    setMapRef(map);
    
    try {
      // If we have data points, try to fit the map to show them all
      if (mapData.points && mapData.points.length > 0) {
        // Create bounds to contain all points
        const bounds = new window.google.maps.LatLngBounds();
        
        // Add all valid points to bounds
        let validPointsAdded = 0;
        mapData.points.forEach(point => {
          if (point && point.latitude && point.longitude) {
            const lat = parseFloat(point.latitude);
            const lng = parseFloat(point.longitude);
            
            if (!isNaN(lat) && !isNaN(lng)) {
              bounds.extend({ lat, lng });
              validPointsAdded++;
            }
          }
        });
        
        // Only adjust bounds if we have valid points
        if (validPointsAdded > 0) {
          map.fitBounds(bounds);
          // Add some padding
          const boundsListener = window.google.maps.event.addListenerOnce(map, 'bounds_changed', () => {
            map.setZoom(Math.min(10, map.getZoom()));
          });
          console.log(`Adjusted map to fit ${validPointsAdded} points`);
        } else {
          console.log("No valid points to adjust bounds");
        }
      } else {
        console.log("No points available to adjust map bounds");
      }
    } catch (error) {
      console.error("Error adjusting map bounds:", error);
    }
  };

  // Update useEffect to use isLoaded
  useEffect(() => {
    console.log("MapView component mounted");
    fetchMapData();
    fetchAvailableSymptoms();
    
    // Add a global error handler for Google Maps
    const handleGoogleMapsError = (error) => {
      console.error("Google Maps error:", error);
      setLoadScriptError(`Google Maps error: ${error.message}`);
      setUseAlternativeDisplay(true);
    };
    
    // Add listener for Google Maps errors
    window.gm_authFailure = () => {
      console.error("Google Maps authentication failed - invalid API key");
      setLoadScriptError("Google Maps authentication failed. Please check your API key.");
      setUseAlternativeDisplay(true);
    };
    
    // If Maps API loaded successfully
    if (isLoaded) {
      setIsApiLoaded(true);
      console.log("Google Maps API fully loaded and ready for use");
    }
    
    // If loading failed
    if (loadError) {
      console.error("Error loading Google Maps API:", loadError);
      setLoadScriptError(loadError.message);
      setUseAlternativeDisplay(true);
    }
    
    return () => {
      // Clean up error handler
      window.gm_authFailure = null;
    };
  }, [isLoaded, loadError]);

  // Restore user location behavior
  useEffect(() => {
    // First, get user location and then fetch data
    getUserLocation();
    
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
  
  // Add a timeout for map loading to fallback to list view if maps doesn't load
  useEffect(() => {
    if (!isLoaded) {
      const mapLoadTimeout = setTimeout(() => {
        if (!isLoaded) {
          console.log("Maps still not loaded after timeout - falling back to list view");
          setUseAlternativeDisplay(true);
        }
      }, 8000); // 8 second timeout
      
      return () => clearTimeout(mapLoadTimeout);
    }
  }, [isLoaded]);

  // Refetch data when symptom filter changes
  useEffect(() => {
    if (userLocation) {
      fetchMapData(userLocation);
    } else {
      fetchMapData();
    }
  }, [selectedSymptom]);

  // Verify Google Maps is rendering
  useEffect(() => {
    // Force map rerender after component is mounted
    if (isLoaded && mapRef) {
      console.log("Forcing map rerender...");
      const timeoutId = setTimeout(() => {
        if (mapRef) {
          // Trigger a resize event to force map to render
          window.dispatchEvent(new Event('resize'));
          console.log("Map should be visible now after resize event");
          
          // Add a test marker to verify map is working
          if (window.google && window.google.maps) {
            const testMarker = new window.google.maps.Marker({
              position: center,
              map: mapRef,
              title: "Test marker at center",
              icon: "https://maps.google.com/mapfiles/ms/icons/red-dot.png"
            });
            console.log("Added test marker at center", center);
          }
        }
      }, 1000);
      
      return () => clearTimeout(timeoutId);
    }
  }, [isLoaded, mapRef]);

  // Helper to offset multiple markers at the same location
  const getOffsetPosition = (originalPosition, index) => {
    // If this is the first marker at this location, don't offset
    if (index === 0) return originalPosition;
    
    // Create small offsets for subsequent markers at the same location
    // This creates a small cluster effect so all markers remain visible
    const offsetAmount = 0.0003 * (index % 8 + 1); // About 30 meters offset
    const offsetAngle = (index % 8) * (Math.PI / 4); // 8 directions around the point
    
    return {
      lat: originalPosition.lat + offsetAmount * Math.sin(offsetAngle),
      lng: originalPosition.lng + offsetAmount * Math.cos(offsetAngle)
    };
  };

  // Get appropriate marker icon based on symptoms
  const getMarkerIconForSymptom = (symptoms) => {
    if (!symptoms || (Array.isArray(symptoms) && symptoms.length === 0)) {
      return "https://maps.google.com/mapfiles/ms/icons/red-dot.png";
    }
    
    // Make sure we're working with an array
    const symptomArray = Array.isArray(symptoms) ? symptoms : [symptoms];
    
    // Look for specific symptoms in priority order
    if (symptomArray.includes('ulcers_on_tongue')) {
      return "https://maps.google.com/mapfiles/ms/icons/purple-dot.png"; // More visible color for ulcers
    } else if (symptomArray.includes('nodal_skin_eruptions')) {
      return "https://maps.google.com/mapfiles/ms/icons/green-dot.png";
    } else if (symptomArray.includes('chills') || symptomArray.includes('fever')) {
      return "https://maps.google.com/mapfiles/ms/icons/red-dot.png";
    } else if (symptomArray.includes('joint_pain')) {
      return "https://maps.google.com/mapfiles/ms/icons/blue-dot.png";
    } else if (symptomArray.includes('vomiting')) {
      return "https://maps.google.com/mapfiles/ms/icons/orange-dot.png";
    }
    
    // Default color
    return "https://maps.google.com/mapfiles/ms/icons/yellow-dot.png";
  };

  // Modify renderMap function for marker creation
  const renderMap = () => {
    if (!isLoaded) {
      return (
        <Box display="flex" justifyContent="center" alignItems="center" height="600px">
          <CircularProgress />
          <Typography variant="body2" sx={{ ml: 2 }}>Loading Google Maps...</Typography>
        </Box>
      );
    }
    
    if (loadError) {
      return (
        <Box display="flex" justifyContent="center" alignItems="center" height="600px">
          <Typography color="error" variant="body1">
            Error loading Google Maps: {loadError.message}
          </Typography>
        </Box>
      );
    }
    
    // Group markers by location to handle offsets
    const groupMarkersForRendering = () => {
      if (!mapData.points || mapData.points.length === 0) return [];
      
      // Group by coordinates
      const locationGroups = {};
      
      mapData.points.forEach(point => {
        if (!point.latitude || !point.longitude) return;
        
        const key = `${point.latitude},${point.longitude}`;
        if (!locationGroups[key]) {
          locationGroups[key] = [];
        }
        locationGroups[key].push(point);
      });
      
      // Flatten with index for offset
      const result = [];
      Object.values(locationGroups).forEach(group => {
        group.forEach((point, groupIndex) => {
          result.push({...point, groupIndex});
        });
      });
      
      return result;
    };
    
    return (
      <div style={{ width: '100%', height: '600px', position: 'relative' }}>
        <GoogleMap
          mapContainerStyle={{
            width: '100%',
            height: '100%',
            position: 'absolute',
            top: 0,
            left: 0
          }}
          center={userLocation || center}
          zoom={5}
          onLoad={onMapLoad}
          options={{
            fullscreenControl: true,
            streetViewControl: true,
            mapTypeControl: true,
            zoomControl: true,
            mapTypeId: window.google?.maps?.MapTypeId?.ROADMAP
          }}
        >
          {/* Only render markers when API is properly loaded */}
          {isLoaded && userLocation && (
            <Marker
              position={{
                lat: parseFloat(userLocation.lat),
                lng: parseFloat(userLocation.lng)
              }}
              icon={{
                url: "https://maps.google.com/mapfiles/ms/icons/blue-dot.png"
              }}
              title="Your Location"
            />
          )}
          
          {/* Render all markers directly - handle overlapping points */}
          {isLoaded && mapData.points && mapData.points.length > 0 && (() => {
            // Group markers by location
            const locationGroups = {};
            
            mapData.points.forEach(point => {
              if (!point.latitude || !point.longitude) return;
              
              const key = `${point.latitude},${point.longitude}`;
              if (!locationGroups[key]) {
                locationGroups[key] = [];
              }
              locationGroups[key].push(point);
            });
            
            return Object.entries(locationGroups).flatMap(([coordKey, points]) => {
              return points.map((point, groupIndex) => {
                // Parse coordinates to ensure they're numbers
                const lat = parseFloat(point.latitude);
                const lng = parseFloat(point.longitude);
                
                if (isNaN(lat) || isNaN(lng)) {
                  console.warn(`Point has invalid coordinates:`, point);
                  return null;
                }
                
                // Calculate offset position if needed (spread out points with same coordinates)
                const position = getOffsetPosition({lat, lng}, groupIndex);
                
                console.log(`Creating marker ${coordKey}-${groupIndex} at ${position.lat},${position.lng} for symptoms:`, point.symptoms);
                
                return (
                  <Marker
                    key={`marker-${coordKey}-${groupIndex}`}
                    position={position}
                    onClick={() => setSelectedPoint(point)}
                    icon={{
                      url: getMarkerIconForSymptom(point.symptoms)
                    }}
                  />
                );
              });
            });
          })()}
          
          {/* Info window for selected marker */}
          {isLoaded && selectedPoint && (
            <InfoWindow
              position={{ 
                lat: parseFloat(selectedPoint.latitude), 
                lng: parseFloat(selectedPoint.longitude) 
              }}
              onCloseClick={() => setSelectedPoint(null)}
            >
              <div style={{ padding: '12px', maxWidth: '300px' }}>
                <Typography variant="subtitle1" sx={{ fontWeight: 'bold', mb: 1, color: '#1976d2' }}>
                  Symptom Report
                </Typography>
                
                {/* Symptoms */}
                <Typography variant="body2" sx={{ mb: 1.5, fontWeight: 'medium' }}>
                  <span style={{ fontWeight: 'bold' }}>Symptoms:</span> {' '}
                  {Array.isArray(selectedPoint.symptoms) 
                    ? selectedPoint.symptoms.map(s => s.replace('_', ' ')).join(', ')
                    : 'No symptoms reported'}
                </Typography>
                
                {/* Disease Prediction */}
                {selectedPoint.diagnosis && (
                  <Typography variant="body2" sx={{ mb: 1, color: selectedPoint.confidence > 70 ? '#d32f2f' : '#ed6c02' }}>
                    <span style={{ fontWeight: 'bold' }}>Predicted Disease:</span> {' '}
                    {selectedPoint.diagnosis} 
                    {selectedPoint.confidence !== undefined && (
                      <span style={{ fontSize: '0.9em', fontStyle: 'italic', ml: 1 }}>
                        ({selectedPoint.confidence.toFixed(1)}% confidence)
                      </span>
                    )}
                  </Typography>
                )}
                
                {/* Location Info */}
                {(selectedPoint.city || selectedPoint.region || selectedPoint.country) && (
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    <span style={{ fontWeight: 'bold' }}>Location:</span> {' '}
                    {[
                      selectedPoint.city, 
                      selectedPoint.region, 
                      selectedPoint.country
                    ].filter(Boolean).join(', ')}
                  </Typography>
                )}
                
                {/* Timestamp */}
                {selectedPoint.timestamp && (
                  <Typography variant="body2" sx={{ fontSize: '0.85em', color: 'text.secondary', mt: 1 }}>
                    Reported: {new Date(selectedPoint.timestamp).toLocaleString()}
                  </Typography>
                )}
                
                {/* Medication Recommendations (if available) */}
                {selectedPoint.medications && selectedPoint.medications.length > 0 && (
                  <>
                    <Typography variant="body2" sx={{ mt: 1.5, mb: 0.5, fontWeight: 'bold' }}>
                      Recommended Medications:
                    </Typography>
                    <Typography variant="body2" component="ul" sx={{ m: 0, pl: 2 }}>
                      {selectedPoint.medications.map((med, index) => (
                        <li key={index}>{med}</li>
                      ))}
                    </Typography>
                  </>
                )}
              </div>
            </InfoWindow>
          )}
        </GoogleMap>
      </div>
    );
  };

  // Add symptom legend component
  const renderSymptomLegend = () => {
    return (
      <Box sx={{ mb: 2 }}>
        <Typography variant="subtitle1" sx={{ mb: 1 }}>Symptom Legend</Typography>
        <Stack direction="row" spacing={1} flexWrap="wrap">
          <Chip 
            label="Fever" 
            size="small"
            sx={{ bgcolor: '#FF5733', color: 'white', m: 0.5 }}
          />
          <Chip 
            label="Cough" 
            size="small"
            sx={{ bgcolor: '#FFAA33', color: 'white', m: 0.5 }}
          />
          <Chip 
            label="Headache" 
            size="small"
            sx={{ bgcolor: '#FF33AA', color: 'white', m: 0.5 }}
          />
          <Chip 
            label="Skin Rash" 
            size="small"
            sx={{ bgcolor: '#33FF57', color: 'white', m: 0.5 }}
          />
          <Chip 
            label="Joint Pain" 
            size="small"
            sx={{ bgcolor: '#FF33FF', color: 'white', m: 0.5 }}
          />
          <Chip 
            label="Nausea/Vomiting" 
            size="small"
            sx={{ bgcolor: '#FF3333', color: 'white', m: 0.5 }}
          />
          <Chip 
            label="Other Symptoms" 
            size="small"
            sx={{ bgcolor: '#808080', color: 'white', m: 0.5 }}
          />
        </Stack>
      </Box>
    );
  };

  // Add alternative display when map isn't available
  const renderAlternativeMapDisplay = () => {
    return (
      <Box sx={{ mt: 2 }}>
        <Typography variant="h6" sx={{ mb: 2 }}>Map Data (List View)</Typography>
        <Box sx={{ mb: 1 }}>
          <Typography variant="body2">
            Showing {mapData.points ? mapData.points.length : 0} data points
          </Typography>
        </Box>
        <Box>
          {mapData.points && mapData.points.length > 0 ? (
            mapData.points.map((point, index) => (
              <Box
                key={index}
                sx={{
                  mb: 2,
                  p: 2,
                  border: '1px solid #eee',
                  borderRadius: 1,
                }}
              >
                <Typography variant="subtitle1">
                  Location {index + 1}
                </Typography>
                <Typography variant="body2">
                  Coordinates: {point.latitude}, {point.longitude}
                </Typography>
                <Box sx={{ mt: 1, mb: 1, display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                  {Array.isArray(point.symptoms) && point.symptoms.map((symptom, i) => (
                    <Chip 
                      key={i} 
                      label={symptom.replace('_', ' ')} 
                      size="small"
                      sx={{ bgcolor: symptomColors[symptom] || defaultColor, color: 'white' }}
                    />
                  ))}
                </Box>
                {point.diagnosis && (
                  <Typography variant="body2">
                    Diagnosis: {point.diagnosis}
                  </Typography>
                )}
              </Box>
            ))
          ) : (
            <Typography variant="body1">No data points available</Typography>
          )}
        </Box>
        
        <Button
          variant="contained"
          color="primary"
          size="small"
          onClick={() => setUseAlternativeDisplay(false)}
          sx={{ mt: 2 }}
        >
          Try Map View Again
        </Button>
      </Box>
    );
  };

  // Add debug panel for development/testing
  const renderDebugPanel = () => {
    return (
      <Box sx={{ mt: 3, p: 2, border: '1px solid #f0f0f0', borderRadius: 1 }}>
        <Typography variant="h6" sx={{ mb: 2 }}>Debug Information</Typography>
        
        <Grid container spacing={2}>
          <Grid item xs={12} md={4}>
            <Typography variant="subtitle2">API Status:</Typography>
            <Typography variant="body2" color={testApiStatus.includes('Error') ? 'error' : 'success'}>
              {testApiStatus || 'Not tested'}
            </Typography>
            <Button 
              variant="outlined" 
              size="small" 
              onClick={() => fetchMapData()} 
              sx={{ mt: 1 }}
            >
              Refresh Data
            </Button>
          </Grid>
          
          <Grid item xs={12} md={4}>
            <Typography variant="subtitle2">Map API Status:</Typography>
            <Typography variant="body2">
              Maps API Loaded: {isLoaded ? 'Yes' : 'No'}
            </Typography>
            <Typography variant="body2">
              Maps API Error: {loadError ? loadError.message : 'None'}
            </Typography>
          </Grid>
          
          <Grid item xs={12} md={4}>
            <Typography variant="subtitle2">User Location:</Typography>
            <Typography variant="body2">
              {userLocation 
                ? `Lat: ${userLocation.lat}, Lng: ${userLocation.lng}` 
                : 'Not available'}
            </Typography>
            <Button 
              variant="outlined" 
              size="small" 
              onClick={getUserLocation} 
              sx={{ mt: 1 }}
            >
              Get Location
            </Button>
          </Grid>
        </Grid>
        
        <Box sx={{ mt: 1 }}>
          <Typography variant="subtitle2">API Response:</Typography>
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
              onClick={() => setDebugMode(!debugMode)}
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
              onChange={(event, newValue) => {
                setSelectedSymptom(newValue);
                if (userLocation) {
                  fetchMapData(userLocation);
                } else {
                  fetchMapData();
                }
              }}
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