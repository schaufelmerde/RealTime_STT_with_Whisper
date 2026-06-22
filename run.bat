@echo off
REM ---------------------------------------------------------------------------
REM  RealTime STT with Whisper - launcher
REM  Starts the Streamlit server (in its own window) and opens the browser UI
REM  once the server is actually accepting connections.
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"

set "PORT=8501"
set "URL=http://localhost:%PORT%"
set "PY=%~dp0venv\Scripts\python.exe"

if not exist "%PY%" (
    echo [ERROR] venv not found at "%PY%".
    echo Create it first:
    echo     python -m venv venv
    echo     venv\Scripts\activate
    echo     pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM Headless so Streamlit doesn't open its own tab - we open the browser
REM ourselves below, once we've confirmed the port is live (avoids a dead tab).
start "RealTime STT - server" "%PY%" -m streamlit run main.py --server.port %PORT% --server.headless true

echo Starting server, waiting for %URL% ...
:waitloop
"%PY%" -c "import socket,sys; s=socket.socket(); s.settimeout(1); sys.exit(0 if s.connect_ex(('127.0.0.1',%PORT%))==0 else 1)" 2>nul
if errorlevel 1 (
    timeout /t 1 /nobreak >nul
    goto waitloop
)

start "" "%URL%"
echo.
echo App running at %URL%
echo Close the "RealTime STT - server" window (or press Ctrl+C in it) to stop.
endlocal
