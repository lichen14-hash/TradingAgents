@echo off
cd /d "%~dp0"
echo.
echo   TradingAgents 分析平台
echo   ========================
echo.
echo   正在启动服务...
echo   浏览器访问: http://localhost:8888
echo   按 Ctrl+C 停止服务
echo.
start http://localhost:8888
python -m uvicorn web.server:app --host 0.0.0.0 --port 8888 --log-level info
pause
