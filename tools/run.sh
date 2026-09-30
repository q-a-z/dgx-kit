#!/usr/bin/env bash
# Run DGX-kit from this folder with full control: start, stop, pull, download, edit and publish.
# Models started outside DGX-kit stay watch-only; DGX-kit never stops or changes them.
# Settings, keys and the password live in ../state, next to this folder, so a new deploy keeps them.
#   tools/run.sh                  run it (on 0.0.0.0 only once an admin password is set)
#   tools/run.sh --set-password   set or change the admin password, then exit
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
cd "$here"
port=${DGXKIT_PORT:-8765}
state=${DGXKIT_STATE_DIR:-$(dirname "$here")/state}
mkdir -p "$state"

if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv 2>/dev/null || { echo "python3-venv is missing; install it or run: pip install --user virtualenv"; exit 1; }
fi
.venv/bin/pip install -q --disable-pip-version-check -e .

if [[ ${1:-} == --set-password ]]; then
  read -r -s -p "Admin password for the dashboard (8+ characters): " pw </dev/tty; echo
  [[ ${#pw} -ge 8 ]] || { echo "too short"; exit 1; }
  PW=$pw .venv/bin/python - "$state/admin.pw" <<'PY'
import os, sys
from dgxkit.auth import hash_password
path = sys.argv[1]
fd = os.open(path + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
os.write(fd, hash_password(os.environ["PW"]).encode()); os.close(fd)
os.replace(path + ".tmp", path)
PY
  echo "Password set. Restart DGX-kit to sign everyone out."
  exit 0
fi

bind=${DGXKIT_BIND:-0.0.0.0}
if [[ $bind != 127.0.0.1 && ! -f $state/admin.pw && ${DGXKIT_NO_PASSWORD:-} != 1 ]]; then
  echo "No admin password yet, so anyone on the network could control this box."
  echo "Set one with: tools/run.sh --set-password   (or run with DGXKIT_BIND=127.0.0.1)"
  exit 1
fi
echo "DGX-kit on http://$(hostname -I 2>/dev/null | awk '{print $1}'):$port (state in $state)"
export DGXKIT_MODEL_PATHS=${DGXKIT_MODEL_PATHS:-$HOME/models:$HOME/.cache/huggingface/hub}
DGXKIT_STATE_DIR="$state" DGXKIT_MODELS_DIR=${DGXKIT_MODELS_DIR:-$(dirname "$here")/models} \
  DGXKIT_PORT=$port DGXKIT_BIND=$bind exec .venv/bin/python -m dgxkit.app
