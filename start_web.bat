@echo off
cd /d "%~dp0"
echo Starting TradingAgents Web Server...
echo Open http://localhost:8888 in your browser
echo.
py -m uvicorn web.server:app --host 0.0.0.0 --port 8888
pause
