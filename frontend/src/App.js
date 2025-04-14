import React, { useState } from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { 
  CssBaseline, Box, AppBar, Toolbar, Typography, Tabs, Tab, 
  Container, Paper, useMediaQuery, IconButton
} from '@mui/material';
import MapView from './components/MapView';
import Statistics from './components/Statistics';
import DiagnosisForm from './components/DiagnosisForm';
import BiotechIcon from '@mui/icons-material/Biotech';

// Enhanced theme with a more professional color palette
const theme = createTheme({
  palette: {
    primary: {
      main: '#2D4059', // Deep blue
      light: '#3A5A7C',
      dark: '#1A2A3A',
      contrastText: '#FFFFFF',
    },
    secondary: {
      main: '#EA5455', // Coral red
      light: '#FF6B6B',
      dark: '#D03E3F',
      contrastText: '#FFFFFF',
    },
    accent: {
      main: '#44CFCB', // Teal
    },
    background: {
      default: '#F8F9FA',
      paper: '#FFFFFF',
    },
    text: {
      primary: '#333333',
      secondary: '#666666',
    },
  },
  typography: {
    fontFamily: '"Poppins", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontWeight: 700,
    },
    h2: {
      fontWeight: 600,
    },
    h3: {
      fontWeight: 600,
    },
    h4: {
      fontWeight: 600,
    },
    h5: {
      fontWeight: 600,
    },
    h6: {
      fontWeight: 600,
    },
    button: {
      textTransform: 'none',
      fontWeight: 500,
    },
  },
  shape: {
    borderRadius: 8,
  },
  components: {
    MuiAppBar: {
      styleOverrides: {
        root: {
          boxShadow: '0 2px 10px rgba(0, 0, 0, 0.08)',
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.05)',
        },
      },
    },
    MuiTab: {
      styleOverrides: {
        root: {
          fontWeight: 500,
          fontSize: '0.95rem',
        },
      },
    },
  },
});

function TabPanel(props) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`tabpanel-${index}`}
      aria-labelledby={`tab-${index}`}
      {...other}
    >
      {value === index && <Box>{children}</Box>}
    </div>
  );
}

function App() {
  const [tabValue, setTabValue] = useState(0);
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
        <AppBar position="static">
          <Toolbar sx={{ py: 1 }}>
            <BiotechIcon sx={{ mr: 1.5, fontSize: 28 }} />
            <Typography 
              variant="h5" 
              component="div" 
              sx={{ 
                flexGrow: 1, 
                fontWeight: 600,
                fontSize: isMobile ? '1.1rem' : '1.5rem',
                letterSpacing: '-0.01em'
              }}
            >
              EpiScope
              <Typography 
                component="span" 
                sx={{ 
                  display: { xs: 'none', sm: 'inline' },
                  ml: 1, 
                  color: 'rgba(255,255,255,0.8)',
                  fontSize: '0.9rem',
                  fontWeight: 400
                }}
              >
                A Real-Time Geo-Analytics Platform for Symptom and Disease Trend Analysis
              </Typography>
            </Typography>
          </Toolbar>
          <Tabs
            value={tabValue}
            onChange={handleTabChange}
            indicatorColor="secondary"
            textColor="inherit"
            variant="fullWidth"
            aria-label="app tabs"
            sx={{ 
              '.MuiTabs-indicator': { 
                height: 3,
                borderTopLeftRadius: 3,
                borderTopRightRadius: 3
              } 
            }}
          >
            <Tab label="Diagnosis" id="tab-0" aria-controls="tabpanel-0" />
            <Tab label="Geo Analysis" id="tab-1" aria-controls="tabpanel-1" />
            <Tab label="Trend Statistics" id="tab-2" aria-controls="tabpanel-2" />
          </Tabs>
        </AppBar>

        <Container maxWidth="xl" sx={{ flexGrow: 1, py: 3 }}>
          <TabPanel value={tabValue} index={0}>
            <Paper elevation={2} sx={{ borderRadius: 2, overflow: 'hidden' }}>
              <DiagnosisForm />
            </Paper>
          </TabPanel>
          <TabPanel value={tabValue} index={1}>
            <Paper elevation={2} sx={{ p: 0, height: 'calc(100vh - 180px)', borderRadius: 2, overflow: 'hidden' }}>
              <MapView />
            </Paper>
          </TabPanel>
          <TabPanel value={tabValue} index={2}>
            <Paper elevation={2} sx={{ borderRadius: 2, overflow: 'hidden' }}>
              <Statistics />
            </Paper>
          </TabPanel>
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default App; 