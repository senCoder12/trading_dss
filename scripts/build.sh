#!/usr/bin/env bash
# scripts/build.sh — Build the Trading DSS for production deployment.
#
# Usage:
#   bash scripts/build.sh           # full build (frontend + validation)
#   bash scripts/build.sh --no-frontend  # skip npm build (Python only)
#
# After a successful build, start the system with:
#   python scripts/run_system.py
#   Dashboard: http://localhost:8000
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
RESET='\033[0m'

ok()   { echo -e "${GREEN}  ✅ $*${RESET}"; }
warn() { echo -e "${YELLOW}  ⚠️  $*${RESET}"; }
fail() { echo -e "${RED}  ❌ $*${RESET}"; exit 1; }
step() { echo -e "\n${YELLOW}[build]${RESET} $*"; }

# ── Argument handling ─────────────────────────────────────────────────────────
BUILD_FRONTEND=true
for arg in "$@"; do
  case "$arg" in
    --no-frontend) BUILD_FRONTEND=false ;;
    -h|--help)
      echo "Usage: bash scripts/build.sh [--no-frontend]"
      exit 0
      ;;
    *)
      warn "Unknown argument: $arg"
      ;;
  esac
done

# ── Header ────────────────────────────────────────────────────────────────────
echo ""
echo "  Trading Decision Support System — Production Build"
echo "  ════════════════════════════════════════════════════"
echo "  Project root: $PROJECT_ROOT"

cd "$PROJECT_ROOT"

# ── Step 1: Python environment check ─────────────────────────────────────────
step "Checking Python environment..."

if ! command -v python3 &>/dev/null; then
  fail "python3 not found. Install Python >= 3.11."
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [[ "$PYTHON_MAJOR" -lt 3 ]] || [[ "$PYTHON_MAJOR" -eq 3 && "$PYTHON_MINOR" -lt 11 ]]; then
  fail "Python $PYTHON_VERSION is too old — need >= 3.11"
fi
ok "Python $PYTHON_VERSION"

# ── Step 2: Python dependencies ───────────────────────────────────────────────
step "Installing Python dependencies..."

if [[ ! -f requirements.txt ]]; then
  fail "requirements.txt not found in $PROJECT_ROOT"
fi

pip install -q -r requirements.txt
ok "Python dependencies installed"

# ── Step 3: Build React frontend ──────────────────────────────────────────────
if [[ "$BUILD_FRONTEND" == true ]]; then
  step "Building React frontend..."

  FRONTEND_DIR="$PROJECT_ROOT/frontend"
  if [[ ! -d "$FRONTEND_DIR" ]]; then
    fail "frontend/ directory not found. Is the repository complete?"
  fi

  if ! command -v node &>/dev/null; then
    fail "node not found. Install Node.js >= 18 to build the frontend."
  fi
  if ! command -v npm &>/dev/null; then
    fail "npm not found. Install Node.js >= 18 to build the frontend."
  fi

  NODE_VERSION=$(node --version)
  ok "Node $NODE_VERSION"

  cd "$FRONTEND_DIR"
  npm install --silent
  npm run build
  cd "$PROJECT_ROOT"

  if [[ -d "$FRONTEND_DIR/dist" ]]; then
    DIST_SIZE=$(du -sh "$FRONTEND_DIR/dist" 2>/dev/null | cut -f1)
    ok "Frontend built → frontend/dist/ ($DIST_SIZE)"
  else
    fail "npm run build completed but frontend/dist/ was not created"
  fi
else
  warn "Skipping frontend build (--no-frontend)"
fi

# ── Step 4: Validate configuration ────────────────────────────────────────────
step "Running configuration validation..."
python3 scripts/validate_config.py
# validate_config.py calls sys.exit(1) on failure — the set -e above will
# propagate that and abort the build.

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "  ════════════════════════════════════════════════════"
ok "Build complete!"
echo ""
echo "  Start the system:"
echo "    python scripts/run_system.py"
echo ""
echo "  Dashboard:   http://localhost:8000"
echo "  API docs:    http://localhost:8000/docs"
echo ""
