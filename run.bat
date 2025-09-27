@echo off
REM Quasar - Radio Astronomy Assistant
REM Windows startup script

echo =========================================
echo    QUASAR - Radio Astronomy Assistant
echo =========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if .env file exists
if not exist ".env" (
    echo Warning: .env file not found
    echo Please ensure your environment variables are configured
    echo.
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/update dependencies
echo Checking dependencies...
pip install -q --upgrade pip
pip install -q -r requirements.txt 2>nul

REM Clear screen for clean start
cls

echo ================================================
echo.
echo      QUASAR IS STARTING UP
echo.
echo      Radio Astronomy Intelligence System
echo.
echo ================================================
echo.
echo Launching Streamlit interface...
echo Opening browser at http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo ----------------------------------------
echo.

REM Run Streamlit app
streamlit run ui/app.py --server.port=8501 --server.address=localhost --browser.serverAddress=localhost --theme.base="dark"
