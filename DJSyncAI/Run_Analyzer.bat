@echo off
cd /d "%~dp0"

echo ========================================================
echo Starting DJSyncAI: AI Smart Track Discovery (Extended Mixes Only)
echo ========================================================

:: Start the Python server in its own window so logs and status are clearly visible
start "DJSyncAI Analyzer Server" python server.py

:: Wait 2 seconds for the server to spin up
timeout /t 2 /nobreak >nul 2>&1 || ping -n 3 127.0.0.1 >nul 2>&1

:: Open default web browser straight to the app
start http://localhost:8000 2>nul || explorer "http://localhost:8000"

echo.
echo DJSyncAI Server is running at http://localhost:8000
echo Logs are captured in: %~dp0Logs\logs.txt
echo ========================================================
