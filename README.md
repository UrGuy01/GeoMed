# GeoMed - Health Analytics Platform

GeoMed is a comprehensive health analytics platform that combines machine learning and AI to provide medical diagnoses based on symptoms, visualize health data on maps, and analyze health statistics.

## Features

- **Symptom-Based Diagnosis**: Users can enter their symptoms and receive diagnoses from both a machine learning model and an AI assistant.
- **Geographic Visualization**: Health data is plotted on a map to show disease spread and patterns in different areas.
- **Statistical Analysis**: Interactive charts and graphs show symptom distributions, disease prevalence, and trends over time.
- **Hybrid Intelligence**: Combines traditional machine learning models with Gemini AI for more accurate diagnoses.

## Project Structure

```
GeoMed/
├── backend/                # Flask backend server
│   └── app.py              # Main server file with API endpoints
├── frontend/               # React frontend
│   ├── src/                # Source code
│   │   ├── components/     # React components
│   │   │   ├── DiagnosisForm.js    # Symptom entry form
│   │   │   ├── MapView.js          # Map visualization
│   │   │   ├── Statistics.js       # Statistical charts
│   │   │   └── DateRangePicker.js  # Date range selector
│   │   └── App.js          # Main React application
├── models/                 # ML model files
│   ├── improved_ensemble_model.pkl # Ensemble ML model
│   ├── disease_mapping.pkl         # Disease label mapping
│   └── symptom_list.pkl            # List of symptoms
├── development/            # Development and research files
├── notebooks/              # Jupyter notebooks for data analysis
└── start_app.bat           # Script to start the application
```

## Technologies Used

- **Backend**: Python, Flask, Supabase
- **Frontend**: React, Material-UI, Recharts
- **AI/ML**: Scikit-learn, Google Gemini API
- **Database**: Supabase (PostgreSQL)
- **Maps**: Google Maps API

## Setup and Installation

### Prerequisites

- Python 3.8+
- Node.js 14+
- Google Maps API key
- Google Gemini API key
- Supabase account and API keys

### Backend Setup

1. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the root directory with the following variables:
   ```
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_key
   GOOGLE_API_KEY=your_gemini_api_key
   ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```
   cd frontend
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Create a `.env.local` file in the frontend directory:
   ```
   REACT_APP_GOOGLE_MAPS_API_KEY=your_google_maps_api_key
   ```

## Running the Application

1. Run the provided batch script:
   ```
   start_app.bat
   ```

Or start the servers manually:

1. Start the backend server:
   ```
   cd backend
   python app.py
   ```

2. Start the frontend development server:
   ```
   cd frontend
   npm start
   ```

3. Open your browser and navigate to http://localhost:3000

## Database Schema

The application uses a Supabase database with the following structure:

**diagnoses table:**
- `id`: Auto-incremented primary key
- `symptoms`: Array of symptom strings
- `ml_diagnosis`: String, diagnosis from ML model
- `ml_confidence`: Float, confidence level of ML diagnosis
- `llm_diagnosis`: String, diagnosis from Gemini AI
- `llm_confidence`: Float, confidence level of AI diagnosis
- `latitude`: Float, location latitude
- `longitude`: Float, location longitude
- `timestamp`: Timestamp of diagnosis

## License

This project is for educational purposes only.

## Environment Variables and Security

### Required Environment Variables

This project uses several environment variables for configuration and security. Create `.env` files in appropriate directories with the following variables:

**Root Directory `.env`:**
```
# Google API Keys
GOOGLE_MAPS_API_KEY=your_google_maps_api_key
GOOGLE_API_KEY=your_gemini_api_key

# Supabase Configuration
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
SUPABASE_SECRET_KEY=your_supabase_secret_key

# IP Info Token
IPINFO_TOKEN=your_ipinfo_token

# React App Environment Variables
REACT_APP_IPINFO_TOKEN=your_ipinfo_token
REACT_APP_API_URL=http://localhost:5000

# Database Configuration
DB_HOST=localhost
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_NAME=geomed
```

**Frontend Directory `.env**:
```
# Google Maps API key
REACT_APP_GOOGLE_MAPS_API_KEY=your_google_maps_api_key

# API URL
REACT_APP_API_URL=http://localhost:5000

# IPInfo token
REACT_APP_IPINFO_TOKEN=your_ipinfo_token
```

### Security Notes

- All `.env` files are included in `.gitignore` to prevent committing sensitive information
- Never commit API keys, passwords, or tokens to version control
- When deploying, use environment variable management systems provided by your hosting platform
- For local development, keep your `.env` files secure and do not share them

## Acknowledgements

- Google for the Gemini API
- Supabase for the backend database
- Various open source libraries and frameworks

