import React, { useState, useEffect } from 'react';
import { 
  Typography, 
  Box, 
  Paper, 
  TextField, 
  Button, 
  Autocomplete, 
  Chip, 
  CircularProgress, 
  Alert,
  Card,
  CardContent,
  Divider,
  List,
  ListItem,
  ListItemText
} from '@mui/material';
import axios from 'axios';

const DiagnosisForm = () => {
  const [symptoms, setSymptoms] = useState([]);
  const [availableSymptoms, setAvailableSymptoms] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);
  const [location, setLocation] = useState(null);
  const [loadingSymptoms, setLoadingSymptoms] = useState(true);
  const [inputValue, setInputValue] = useState('');
  const [locationStatus, setLocationStatus] = useState('');

  // Fetch symptom list on component mount
  useEffect(() => {
    const fetchSymptoms = async () => {
      try {
        setLoadingSymptoms(true);
        const response = await axios.get('http://localhost:5000/api/symptoms');
        setAvailableSymptoms(response.data.symptoms);
      } catch (err) {
        console.error('Error fetching symptoms:', err);
        setError('Failed to load available symptoms. Please try again later.');
      } finally {
        setLoadingSymptoms(false);
      }
    };

    // Get user's location with browser geolocation API
    const getLocation = async () => {
      setLocationStatus('Detecting your location...');
      
      // Try browser geolocation first
      if (navigator.geolocation) {
        try {
          navigator.geolocation.getCurrentPosition(
            (position) => {
              const coords = {
                latitude: position.coords.latitude,
                longitude: position.coords.longitude,
                accuracy: position.coords.accuracy,
                source: 'browser'
              };
              setLocation(coords);
              setLocationStatus('Location detected');
              console.log('Location detected:', coords);
            },
            async (err) => {
              console.warn('Geolocation error:', err);
              setLocationStatus('Browser location access denied, using IP location...');
              
              // Fallback to IP geolocation
              try {
                const ipInfoToken = process.env.REACT_APP_IPINFO_TOKEN || '';
                const response = await axios.get(`https://ipinfo.io/json${ipInfoToken ? `?token=${ipInfoToken}` : ''}`);
                if (response.data && response.data.loc) {
                  const [latitude, longitude] = response.data.loc.split(',').map(parseFloat);
                  const coords = {
                    latitude,
                    longitude,
                    accuracy: 5000, // IP geolocation is less accurate, ~city level
                    source: 'ip',
                    ip: response.data.ip,
                    city: response.data.city,
                    region: response.data.region,
                    country: response.data.country
                  };
                  setLocation(coords);
                  setLocationStatus('Using approximate location');
                  console.log('IP location detected:', coords);
                } else {
                  setLocationStatus('Location detection failed');
                }
              } catch (ipError) {
                console.error('IP geolocation error:', ipError);
                setLocationStatus('Location detection failed');
              }
            },
            { enableHighAccuracy: true, timeout: 5000, maximumAge: 0 }
          );
        } catch (geoError) {
          console.error('Geolocation error:', geoError);
          setLocationStatus('Location detection failed');
        }
      } else {
        setLocationStatus('Geolocation not supported by your browser');
      }
    };

    fetchSymptoms();
    getLocation();
  }, []);

  const handleAddSymptom = (event, value) => {
    if (value && !symptoms.includes(value)) {
      setSymptoms([...symptoms, value]);
      setInputValue('');
    }
  };

  const handleRemoveSymptom = (symptomToRemove) => {
    setSymptoms(symptoms.filter(symptom => symptom !== symptomToRemove));
  };

  const handleAddSuggestedSymptom = (symptom) => {
    if (!symptoms.includes(symptom)) {
      setSymptoms([...symptoms, symptom]);
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    
    if (symptoms.length === 0) {
      setError('Please select at least one symptom');
      return;
    }

    setLoading(true);
    setError(null);
    
    try {
      // Prepare payload with symptoms and location data
      const payload = {
        symptoms,
        location: location || null,
        timestamp: new Date().toISOString()
      };
      
      console.log('Submitting diagnosis with data:', payload);
      
      const response = await axios.post('http://localhost:5000/api/diagnose', payload, {
        timeout: 30000 // 30 second timeout
      });
      
      setResults(response.data);
    } catch (err) {
      console.error('Error during diagnosis:', err);
      setError('Failed to get diagnosis. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Symptom Diagnosis
      </Typography>
      
      <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
        <Typography variant="h6" gutterBottom>
          Enter Your Symptoms
        </Typography>
        
        <form onSubmit={handleSubmit}>
          <Box sx={{ mb: 3 }}>
            <Autocomplete
              options={availableSymptoms || []}
              loading={loadingSymptoms}
              value={null}
              inputValue={inputValue}
              onInputChange={(event, newInputValue) => {
                setInputValue(newInputValue);
              }}
              onChange={handleAddSymptom}
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Search symptoms"
                  variant="outlined"
                  fullWidth
                  InputProps={{
                    ...params.InputProps,
                    endAdornment: (
                      <>
                        {loadingSymptoms ? <CircularProgress color="inherit" size={20} /> : null}
                        {params.InputProps.endAdornment}
                      </>
                    ),
                  }}
                />
              )}
              disabled={loading}
            />
          </Box>
          
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 3 }}>
            {symptoms.map((symptom) => (
              <Chip
                key={symptom}
                label={symptom.replace('_', ' ')}
                onDelete={() => handleRemoveSymptom(symptom)}
                color="primary"
              />
            ))}
            {symptoms.length === 0 && (
              <Typography variant="body2" color="text.secondary">
                No symptoms selected. Please search and select your symptoms above.
              </Typography>
            )}
          </Box>
          
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
            <Typography variant="body2" sx={{ mr: 1 }}>
              Location Status:
            </Typography>
            <Typography 
              variant="body2" 
              color={
                locationStatus.includes('detected') || locationStatus.includes('approximate') 
                  ? 'success.main' 
                  : locationStatus.includes('Detecting') 
                    ? 'info.main' 
                    : 'warning.main'
              }
            >
              {locationStatus || 'Unknown'}
            </Typography>
            {location && (
              <Chip 
                size="small" 
                label={location.source === 'browser' ? 'Precise' : 'Approximate'} 
                color={location.source === 'browser' ? 'success' : 'warning'}
                sx={{ ml: 1 }}
              />
            )}
          </Box>
          
          {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
          
          <Button
            type="submit"
            variant="contained"
            color="primary"
            disabled={loading || symptoms.length === 0}
            sx={{ mt: 2 }}
          >
            {loading ? <CircularProgress size={24} color="inherit" /> : 'Get Diagnosis'}
          </Button>
        </form>
      </Paper>
      
      {results && (
        <Card elevation={3}>
          <CardContent>
            <Typography variant="h5" gutterBottom>
              Diagnosis Results
            </Typography>
            
            <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, gap: 2 }}>
              <Card variant="outlined" sx={{ flex: 1, mb: 2 }}>
                <CardContent>
                  <Typography variant="h6" color="primary">
                    {results.source} Diagnosis
                  </Typography>
                  <Typography variant="h5" sx={{ my: 1 }}>
                    {results.diagnosis}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Confidence: {results.confidence.toFixed(1)}%
                  </Typography>
                </CardContent>
              </Card>
            </Box>
            
            {results.needs_more_info && (
              <>
                <Alert severity="info" sx={{ mt: 2, mb: 3 }}>
                  Low confidence diagnosis. Please consider checking for additional symptoms.
                </Alert>
                
                <Box sx={{ mb: 3 }}>
                  <Typography variant="h6" gutterBottom>
                    Suggested Symptoms to Check
                  </Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                    {results.suggested_symptoms.map((symptom) => (
                      <Chip
                        key={symptom}
                        label={symptom.replace('_', ' ')}
                        onClick={() => handleAddSuggestedSymptom(symptom)}
                        color="secondary"
                        variant={symptoms.includes(symptom) ? "filled" : "outlined"}
                        sx={{ cursor: 'pointer' }}
                      />
                    ))}
                  </Box>
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                    Click a symptom to add it to your list and get a more accurate diagnosis.
                  </Typography>
                </Box>
                
                <Box sx={{ mb: 3 }}>
                  <Typography variant="h6" gutterBottom>
                    Alternative Possible Diagnoses
                  </Typography>
                  <List>
                    {results.alternative_diagnoses.map((alt, index) => (
                      <ListItem key={index} divider={index < results.alternative_diagnoses.length - 1}>
                        <ListItemText 
                          primary={alt.disease}
                          secondary={`Key symptoms: ${alt.key_symptoms}`}
                        />
                      </ListItem>
                    ))}
                  </List>
                </Box>
              </>
            )}
            
            <Divider sx={{ my: 2 }} />
            
            <Typography variant="h6" gutterBottom>
              Explanation
            </Typography>
            <Typography variant="body1" paragraph>
              {results.explanation}
            </Typography>
            
            {results.recommendations && (
              <>
                <Divider sx={{ my: 2 }} />
                
                <Typography variant="h6" gutterBottom>
                  Medical Recommendations
                </Typography>
                
                {results.recommendations.description && (
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="subtitle1" gutterBottom>
                      About this condition:
                    </Typography>
                    <Typography variant="body1" paragraph>
                      {results.recommendations.description}
                    </Typography>
                  </Box>
                )}
                
                {results.recommendations.precautions && results.recommendations.precautions.length > 0 && (
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="subtitle1" gutterBottom>
                      Precautions:
                    </Typography>
                    <List dense>
                      {results.recommendations.precautions.map((precaution, index) => (
                        <ListItem key={index}>
                          <ListItemText primary={`• ${precaution}`} />
                        </ListItem>
                      ))}
                    </List>
                  </Box>
                )}
                
                {results.recommendations.medications && (
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="subtitle1" gutterBottom>
                      Medications:
                    </Typography>
                    <Typography variant="body1" paragraph>
                      {results.recommendations.medications}
                    </Typography>
                  </Box>
                )}
                
                {results.recommendations.diet && (
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="subtitle1" gutterBottom>
                      Diet Recommendations:
                    </Typography>
                    <Typography variant="body1" paragraph>
                      {results.recommendations.diet}
                    </Typography>
                  </Box>
                )}
              </>
            )}
            
            <Divider sx={{ my: 2 }} />
            
            <Typography variant="h6" gutterBottom>
              Symptoms Analyzed
            </Typography>
            <List dense>
              {symptoms.map((symptom) => (
                <ListItem key={symptom}>
                  <ListItemText primary={symptom.replace('_', ' ')} />
                </ListItem>
              ))}
            </List>
            
            <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
              Note: This is not a substitute for professional medical advice. Please consult a healthcare provider for proper diagnosis.
            </Typography>
          </CardContent>
        </Card>
      )}
    </Box>
  );
};

export default DiagnosisForm; 