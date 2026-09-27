#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
case "$(uname -s)" in
  Linux*|Darwin*) VENV_PY="$REPO_ROOT/.venv/bin/python" ;;
  MINGW*|MSYS*|CYGWIN*) VENV_PY="$REPO_ROOT/.venv/Scripts/python.exe" ;;
  *) echo "Unsupported shell/OS; use Python 3.12 scripts/bootstrap.py" >&2; exit 1 ;;
esac
if [[ -d "$REPO_ROOT/.venv" ]]; then
  if [[ ! -f "$REPO_ROOT/.venv/pyvenv.cfg" || ! -x "$VENV_PY" ]]; then
    echo "Existing .venv is invalid or belongs to a different OS; move it aside manually." >&2
    exit 1
  fi
  "$VENV_PY" scripts/bootstrap.py "$@"
else
  FOUND=""
  for candidate in python3.12 python3.11 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; assert sys.version_info[:2] in ((3,12),(3,11))' 2>/dev/null; then
      FOUND="$candidate"
      break
    fi
  done
  if [[ -z "$FOUND" ]]; then
    echo "Python 3.12/3.11 with venv is required. Install it yourself, then retry." >&2
    exit 1
  fi
  "$FOUND" scripts/bootstrap.py "$@"
fi
echo "Activate in your own shell: source .venv/bin/activate (Linux/WSL2)"
