@echo off
title NEXUS — Media Recommendation Engine
color 0A
echo.
echo  ███╗   ██╗███████╗██╗  ██╗██╗   ██╗███████╗
echo  ████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝
echo  ██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗
echo  ██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║
echo  ██║ ╚████║███████╗██╔╝ ██╗╚██████╔╝███████║
echo  ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝
echo.
echo  [ Screen Media Recommendation Engine ]
echo  [ Powered by FastAPI + TF-IDF + APScheduler ]
echo.
echo  Starting server on http://localhost:8000 ...
echo  Opening browser in 2 seconds...
echo.

:: Start the server in this window
start "" cmd /c "cd /d "%~dp0" && python main.py"

:: Wait 2 seconds then open the browser to the correct localhost address
timeout /t 2 /nobreak >nul
start "" "http://localhost:8000"

echo  Server launched. Close this window to stop.
echo  (If browser does not open, go to http://localhost:8000 manually)
echo.
pause
