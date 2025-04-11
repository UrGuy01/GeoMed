import { useState, useEffect } from 'react';
import axios from 'axios';
import {
  Container,
  Typography,
  Box,
  Paper,
  CircularProgress,
  Button,
  List,
  ListItem,
  ListItemText,
  Divider,
  Chip,
  Alert,
  Card,
  CardContent,
  Grid
} from '@mui/material';
import Select from 'react-select';

// Define types
interface Symptom {
  value: string;
  label: string;
}

interface PredictionResult {
  predicted_disease: string;
  confidence: number;
  top_predictions: {
    disease: string;
    confidence: number;
  }[];
}

interface Recommendation {
  description: string;
  precautions: string[];
  medications: string;
  diet: string;
}

interface ApiResponse {
  prediction: PredictionResult;
  recommendations: Recommendation;
}

export default function Home() {
  const [symptoms, setSymptoms] = useState<Symptom[]>([]);
  const [selectedSymptoms, setSelectedSymptoms] = useState<Symptom[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<ApiResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Fetch the list of available symptoms when the component mounts
    const fetchSymptoms = async () => {
      try {
        // In a real scenario, this would be the endpoint of your Flask API
        // For demo purposes, we'll simulate with sample data
        
        // Uncomment this line to fetch from your real API
        // const response = await axios.get('http://localhost:5000/api/symptoms');
        
        // Sample data for demonstration
        const sampleSymptoms = [
          'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing',
          'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity',
          'ulcers_on_tongue', 'vomiting', 'burning_micturition', 'fatigue',
          'high_fever', 'headache', 'yellowish_skin', 'dark_urine', 'nausea',
          'loss_of_appetite', 'back_pain', 'constipation', 'abdominal_pain',
          'diarrhoea', 'mild_fever', 'cough', 'chest_pain'
        ];
        
        const formattedSymptoms = sampleSymptoms.map(symptom => ({
          value: symptom,
          label: symptom.replace(/_/g, ' ')
        }));
        
        setSymptoms(formattedSymptoms);
      } catch (err) {
        console.error('Error fetching symptoms:', err);
        setError('Failed to load symptoms. Please try again later.');
      }
    };

    fetchSymptoms();
  }, []);

  const handleSubmit = async () => {
    if (selectedSymptoms.length === 0) {
      setError('Please select at least one symptom');
      return;
    }

    setLoading(true);
    setError(null);
    
    try {
      // Extract just the values from the selected symptoms
      const symptomValues = selectedSymptoms.map(symptom => symptom.value);
      
      // In a real application, this would be your API endpoint
      // const response = await axios.post('http://localhost:5000/api/predict', {
      //   symptoms: symptomValues
      // });
      
      // For demo purposes, simulate an API response
      // Simulate a delayed response
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      // Sample mock response
      const mockResponse = {
        prediction: {
          predicted_disease: 'Common Cold',
          confidence: 85.6,
          top_predictions: [
            { disease: 'Common Cold', confidence: 85.6 },
            { disease: 'Influenza', confidence: 10.2 },
            { disease: 'Allergic Rhinitis', confidence: 4.2 }
          ]
        },
        recommendations: {
          description: "The common cold is a viral infection of your nose and throat (upper respiratory tract). It's usually harmless, although it might not feel that way.",
          precautions: [
            'Rest and stay hydrated',
            'Use over-the-counter medications as needed',
            'Wash hands frequently',
            'Avoid close contact with others'
          ],
          medications: 'Acetaminophen, Decongestants, Cough suppressants',
          diet: 'Consume warm liquids, chicken soup, foods high in vitamin C, and stay well hydrated'
        }
      };
      
      setResult(mockResponse);
    } catch (err) {
      console.error('Error making prediction:', err);
      setError('Failed to get prediction. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="md">
      <Box sx={{ my: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom align="center">
          GeoMed Disease Predictor
        </Typography>
        <Typography variant="subtitle1" align="center" color="text.secondary" paragraph>
          Select your symptoms below to get a disease prediction and medical recommendations
        </Typography>

        {error && (
          <Alert severity="error" sx={{ my: 2 }}>
            {error}
          </Alert>
        )}

        <Paper elevation={3} sx={{ p: 3, mt: 4 }}>
          <Typography variant="h5" gutterBottom>
            Select Your Symptoms
          </Typography>
          <Select
            isMulti
            options={symptoms}
            value={selectedSymptoms}
            onChange={(selected) => setSelectedSymptoms(selected as Symptom[])}
            placeholder="Type or select symptoms..."
            isSearchable
            closeMenuOnSelect={false}
          />

          <Box sx={{ mt: 3, textAlign: 'center' }}>
            <Button
              variant="contained"
              color="primary"
              size="large"
              onClick={handleSubmit}
              disabled={loading || selectedSymptoms.length === 0}
            >
              {loading ? <CircularProgress size={24} /> : 'Get Prediction'}
            </Button>
          </Box>
        </Paper>

        {result && (
          <Box sx={{ mt: 4 }}>
            <Paper elevation={3} sx={{ p: 3 }}>
              <Typography variant="h5" gutterBottom>
                Prediction Results
              </Typography>
              
              <Box sx={{ my: 2, p: 2, bgcolor: 'background.paper', borderRadius: 1 }}>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={6}>
                    <Typography variant="h6">
                      Predicted Disease:
                    </Typography>
                    <Typography variant="h4" color="primary" gutterBottom>
                      {result.prediction.predicted_disease}
                    </Typography>
                    <Chip 
                      label={`Confidence: ${result.prediction.confidence}%`} 
                      color={result.prediction.confidence > 70 ? "success" : "warning"}
                      sx={{ mt: 1 }}
                    />
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <Typography variant="h6">
                      Other Possibilities:
                    </Typography>
                    <List dense>
                      {result.prediction.top_predictions.slice(1).map((pred, index) => (
                        <ListItem key={index}>
                          <ListItemText 
                            primary={pred.disease} 
                            secondary={`Confidence: ${pred.confidence}%`} 
                          />
                        </ListItem>
                      ))}
                    </List>
                  </Grid>
                </Grid>
              </Box>
              
              <Divider sx={{ my: 3 }} />
              
              <Typography variant="h5" gutterBottom>
                Medical Recommendations
              </Typography>
              
              <Card sx={{ mb: 2 }}>
                <CardContent>
                  <Typography variant="h6" color="primary" gutterBottom>
                    Description
                  </Typography>
                  <Typography variant="body1">
                    {result.recommendations.description}
                  </Typography>
                </CardContent>
              </Card>
              
              <Card sx={{ mb: 2 }}>
                <CardContent>
                  <Typography variant="h6" color="primary" gutterBottom>
                    Precautions
                  </Typography>
                  <List>
                    {result.recommendations.precautions.map((precaution, index) => (
                      <ListItem key={index}>
                        <ListItemText primary={`${index + 1}. ${precaution}`} />
                      </ListItem>
                    ))}
                  </List>
                </CardContent>
              </Card>
              
              <Card sx={{ mb: 2 }}>
                <CardContent>
                  <Typography variant="h6" color="primary" gutterBottom>
                    Recommended Medications
                  </Typography>
                  <Typography variant="body1">
                    {result.recommendations.medications}
                  </Typography>
                </CardContent>
              </Card>
              
              <Card>
                <CardContent>
                  <Typography variant="h6" color="primary" gutterBottom>
                    Diet Recommendations
                  </Typography>
                  <Typography variant="body1">
                    {result.recommendations.diet}
                  </Typography>
                </CardContent>
              </Card>
            </Paper>
          </Box>
        )}
      </Box>
    </Container>
  );
} 