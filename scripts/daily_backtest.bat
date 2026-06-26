@echo off
cd /d "d:\rx_aitest\ai_trading\TradingAgents"
if not exist logs mkdir logs
echo [%date% %time%] Starting daily backtest >> logs\daily_backtest.log
py -m tradingagents.backtest.daily_runner >> logs\daily_backtest.log 2>&1
echo [%date% %time%] Completed >> logs\daily_backtest.log
