#!/usr/bin/env bash
# Run DGX-kit from this folder in read-only mode, e.g. on a Spark that already serves models.
# Everything it writes stays in this folder (.venv) and ../state. It reads NVML, /proc, /sys and
# Docker's container list, and refuses to start, stop, pull, build, download or publish.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
cd "$here"
port=${DGXKIT_PORT:-8765}

if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv 2>/dev/null || { echo "python3-venv is missing; install it or run: pip install --user virtualenv"; exit 1; }
fi
.venv/bin/pip install -q --disable-pip-version-check -e .
[[ -f web/dist/index.html ]] || echo "note: web/dist is missing, so only the API is served (build it with npm in web/)"

mkdir -p "$(dirname "$here")/state"
# No admin password in this mode, so only this machine can reach it unless DGXKIT_BIND says otherwise.
bind=${DGXKIT_BIND:-127.0.0.1}
if [[ $bind == 127.0.0.1 ]]; then
  echo "DGX-kit (read-only) on http://127.0.0.1:$port"
  echo "From your laptop: ssh -L $port:127.0.0.1:$port $(id -un)@$(hostname -I 2>/dev/null | awk '{print $1}'), then open http://localhost:$port"
else
  echo "DGX-kit (read-only) on http://$(hostname -I 2>/dev/null | awk '{print $1}'):$port"
fi
# Folders to look for models in (changeable later on the Library tab).
export DGXKIT_MODEL_PATHS=${DGXKIT_MODEL_PATHS:-$HOME/models:$HOME/.cache/huggingface/hub}
DGXKIT_READONLY=1 DGXKIT_STATE_DIR="${DGXKIT_STATE_DIR:-$(dirname "$here")/state}" DGXKIT_MODELS_DIR="$here/state/models-readonly" \
  DGXKIT_PORT=$port DGXKIT_BIND=$bind exec .venv/bin/python -m dgxkit.app
