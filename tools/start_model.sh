#!/usr/bin/env bash
# Start a model through the DGX-kit API, then show its container state and the errors and warnings in its log.
# The session cookie comes from the environment, never from this file:
#   DGXKIT_COOKIE='<dgxkit_session value>' tools/start_model.sh nemotron-3.5-abl-coder-trt [host]
set -euo pipefail
name=${1:?usage: DGXKIT_COOKIE=... start_model.sh MODEL_NAME [host]}
host=${2:-10.1.0.144}
: "${DGXKIT_COOKIE:?set DGXKIT_COOKIE to the dgxkit_session value}"

echo "== start $name"
curl -s -X POST -b "dgxkit_session=$DGXKIT_COOKIE" "http://$host:3000/api/models/$name/start"; echo
echo "== waiting 90s for the engine to load"
sleep 90
echo "== container"
ssh -o BatchMode=yes "q@$host" "docker ps -a --filter name=dgxkit-$name --format '{{.Names}} | {{.Status}}'"
echo "== errors and warnings"
ssh -o BatchMode=yes "q@$host" "docker logs dgxkit-$name 2>&1 | grep -E '\[E\]|\[W\]|Error|Traceback' | cut -c1-300 | sort -u | head -40"
echo "== tail"
ssh -o BatchMode=yes "q@$host" "docker logs --tail 15 dgxkit-$name 2>&1 | grep -v '^Loading' | cut -c1-260"
