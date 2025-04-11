import React, { useState } from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline, Box, AppBar, Toolbar, Typography, Tabs, Tab, Container, Paper } from '@mui/material';
import MapView from './components/MapView';
import Statistics from './components/Statistics';
import DiagnosisForm from './components/DiagnosisForm';

const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#f5f5f5',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h4: {
      fontWeight: 600,
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

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
        <AppBar position="static">
          <Toolbar>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              GeoMed Health Analytics
            </Typography>
          </Toolbar>
          <Tabs
            value={tabValue}
            onChange={handleTabChange}
            indicatorColor="secondary"
            textColor="inherit"
            variant="fullWidth"
            aria-label="app tabs"
          >
            <Tab label="Diagnosis" id="tab-0" aria-controls="tabpanel-0" />
            <Tab label="Map View" id="tab-1" aria-controls="tabpanel-1" />
            <Tab label="Statistics" id="tab-2" aria-controls="tabpanel-2" />
          </Tabs>
        </AppBar>

        <Container maxWidth="xl" sx={{ flexGrow: 1, py: 2 }}>
          <TabPanel value={tabValue} index={0}>
            <Paper elevation={3}>
              <DiagnosisForm />
            </Paper>
          </TabPanel>
          <TabPanel value={tabValue} index={1}>
            <Paper elevation={3} sx={{ p: 2, height: 'calc(100vh - 150px)' }}>
              <MapView />
            </Paper>
          </TabPanel>
          <TabPanel value={tabValue} index={2}>
            <Paper elevation={3}>
              <Statistics />
            </Paper>
          </TabPanel>
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default App; 