@echo off
REM Trading DSS Unified Launcher
REM Starts all 9 services + frontend in parallel (Windows)

setlocal enabledelayedexpansion

cd /d "%~dp0"

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] python not found. Install Python 3.11+ and add to PATH
    exit /b 1
)

REM Check npm
npm --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] npm not found. Install Node.js and add to PATH
    exit /b 1
)

REM Create logs directory
if not exist "logs" mkdir logs

echo.
echo ========================================
echo  Trading DSS ^- Unified Launcher
echo ========================================
echo.

echo [1/2] Starting backend system...
start "Trading DSS Backend" cmd /k "title Backend - Trading DSS & python scripts/run_system.py > logs/trading_dss.log 2>&1"
echo [OK] Backend started (check window)

echo [2/2] Starting frontend...
start "Trading DSS Frontend" cmd /k "cd /d frontend && title Frontend - Trading DSS & npm run dev > ..\logs/frontend.log 2>&1"
echo [OK] Frontend started (check window)

echo.
echo ========================================
echo  All Services Running
echo ========================================
echo.
echo   Backend:  http://localhost:8000
echo   Frontend: http://localhost:3000
echo   API Docs: http://localhost:8000/docs
echo.
echo   Logs: logs\trading_dss.log (backend)
echo         logs\frontend.log (frontend)
echo.
echo   Close both windows to stop all services
echo ========================================
echo.

timeout /t 3
