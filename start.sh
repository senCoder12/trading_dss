#!/usr/bin/env bash
# Trading DSS Unified Launcher
# Starts all 9 services + frontend in parallel

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    kill %1 %2 2>/dev/null || true
    wait
    echo "✅ Services stopped"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RESET='\033[0m'

echo -e "${GREEN}╔════════════════════════════════════════════╗${RESET}"
echo -e "${GREEN}║  Trading DSS — Unified Launcher            ║${RESET}"
echo -e "${GREEN}╚════════════════════════════════════════════╝${RESET}"
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
    echo -e "${RED}❌ python3 not found${RESET}"
    exit 1
fi

# Check Node/npm
if ! command -v npm &>/dev/null; then
    echo -e "${RED}❌ npm not found${RESET}"
    exit 1
fi

# Create logs dir
mkdir -p logs

echo -e "${YELLOW}[1/2]${RESET} Starting backend system..."
python3 scripts/run_system.py > logs/trading_dss.log 2>&1 &
BACKEND_PID=$!
echo -e "${GREEN}✅${RESET} Backend started (PID: $BACKEND_PID)"

echo -e "${YELLOW}[2/2]${RESET} Starting frontend..."
(cd frontend && npm run dev > ../logs/frontend.log 2>&1) &
FRONTEND_PID=$!
echo -e "${GREEN}✅${RESET} Frontend started (PID: $FRONTEND_PID)"

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════╗${RESET}"
echo -e "${GREEN}║  All Services Running                      ║${RESET}"
echo -e "${GREEN}╠════════════════════════════════════════════╣${RESET}"
echo "  📡 Backend:  http://localhost:8000"
echo "  🎨 Frontend: http://localhost:3000"
echo "  📋 API Docs: http://localhost:8000/docs"
echo ""
echo "  Logs: logs/trading_dss.log (backend)"
echo "        logs/frontend.log (frontend)"
echo ""
echo "  Press Ctrl+C to stop all services"
echo -e "${GREEN}╚════════════════════════════════════════════╝${RESET}"
echo ""

# Wait for background jobs
wait
