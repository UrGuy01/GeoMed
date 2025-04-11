@echo off
echo Starting GeoMed Application...

echo Installing Python dependencies...
pip install flask flask-cors python-dotenv numpy pandas scikit-learn google-generativeai supabase

echo Installing npm dependencies...
cd frontend
call npm install
cd ..

echo Starting Backend Server...
start cmd /k "cd backend && python app.py"

echo Starting Frontend Server...
start cmd /k "cd frontend && npm start"

echo Both servers are now running!
echo Backend: http://localhost:5000
echo Frontend: http://localhost:3000

echo Press any key to stop all servers...
pause > nul

echo Stopping servers...
taskkill /f /im python.exe
taskkill /f /im node.exe

echo GeoMed Application stopped. 