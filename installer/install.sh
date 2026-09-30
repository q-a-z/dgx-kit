#!/usr/bin/env bash
# DGX-kit installer: takes a DGX Spark from fresh to a running dashboard.
#
#   install.sh            check, ask a few questions, install what's missing, start DGX-kit (needs sudo)
#   install.sh --user     the same without sudo: a user service, files under your home. Needs Docker
#                         and the NVIDIA container toolkit already set up, and you in the docker group.
#   install.sh --check    read-only: report what's present and what would be installed, change nothing
#   install.sh --dry-run  ask the questions and print every command instead of running it
#   install.sh --updatepath FILE.tar.gz  update the installed DGX-kit from a package on this machine: unpacks it,
#                         rebuilds the image, restarts the service. Settings, keys, recipes and password are kept,
#                         and the previous image stays as :previous for a rollback. Add --user for a user install.
#   install.sh --updaterepo URL  the same, fetching the latest from a git repository
#   install.sh --update   the same from the package this script sits in (an already unpacked one)
#   An update needs no sudo password unless the service file itself changes.
#
# The install also leaves a command in ~/.local/bin:   dgx-kit update [PACKAGE.tar.gz]   dgx-kit version
#   install.sh --uninstall  stop and remove the service and the dashboard; asks before touching anything else.
#                         Your models and the compiled-kernel caches are never deleted. Add --user for a user install.
#
# Flags combine (--user --check). Each step checks first and skips what's already there, so re-running is safe.
#
# Every question can be answered ahead of time for an unattended install, with DGXKIT_INSTALL_<NAME>:
#   MODELS_DIR PORT BIND ADMIN_PASSWORD HF_TOKEN GATEWAY_PORT PULL_NOW UPGRADE READONLY SERVICE
# and for --uninstall: REMOVE_MODELS REMOVE_DATA YES
set -euo pipefail

MODE=install
SCOPE=system
ACTION=install
UPDATE_FILE=""
UPDATE_REPO=""
DEFAULT_UPDATE_REPO=${DGXKIT_UPDATE_REPO:-https://github.com/AIPossum/dgx-kit.git}  # where "dgx-kit update" fetches from
usage() { echo "usage: $0 [--user] [--check|--dry-run] [--uninstall | --update | --updatepath FILE.tar.gz | --updaterepo URL]" >&2; exit 2; }
while (( $# )); do
  case $1 in
    --check) MODE=check ;;
    --dry-run) MODE=dry ;;
    --user) SCOPE=user ;;
    --uninstall) ACTION=uninstall ;;
    --update) ACTION=update ;;
    --updatepath) ACTION=update; UPDATE_FILE=${2:-}; [[ -n $UPDATE_FILE ]] || usage; shift ;;
    --updatepath=*) ACTION=update; UPDATE_FILE=${1#*=} ;;
    --updaterepo) ACTION=update; UPDATE_REPO=${2:-}; [[ -n $UPDATE_REPO ]] || usage; shift ;;
    --updaterepo=*) ACTION=update; UPDATE_REPO=${1#*=} ;;
    *) usage ;;
  esac
  shift
done
[[ $ACTION == uninstall && $MODE == check ]] && MODE=dry  # nothing to only check when removing: show the commands instead

SERVICE=${DGXKIT_INSTALL_SERVICE:-dgx-kit}  # names the service, the container and the folders
[[ $SERVICE =~ ^[a-z0-9][a-z0-9-]*$ ]] || { echo "DGXKIT_INSTALL_SERVICE must be lower-case letters, digits and dashes" >&2; exit 2; }
IMAGE=${DGXKIT_IMAGE:-dgx-kit:latest}
USER=${USER:-$(id -un)}
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)  # the checkout this script came from; the image is built from it
CACHE_DIR=$HOME/.cache  # kernels the engines compile live here on the host, shared by every model container

if [[ $SCOPE == system ]]; then
  CONF_DIR=/etc/$SERVICE
  STATE_DIR=/var/lib/$SERVICE
  UNIT_DIR=/etc/systemd/system
  WANTED_BY=multi-user.target
  SUDO=(sudo)          # what writes system files
  SUDO_N=(sudo -n)     # the same, but never asks for a password (reading an old key)
  CTL=(systemctl)
else
  CONF_DIR=$HOME/.config/$SERVICE
  STATE_DIR=$HOME/.local/share/$SERVICE
  UNIT_DIR=$HOME/.config/systemd/user
  WANTED_BY=default.target
  SUDO=()
  SUDO_N=()
  CTL=(systemctl --user)
fi

ok()   { printf '  \033[32mok\033[0m    %s\n' "$*"; }
miss() { printf '  \033[33mneeds\033[0m %s\n' "$*"; }
bad()  { printf '  \033[31mstop\033[0m  %s\n' "$*"; }
run()  { if [[ $MODE == install ]]; then "$@"; else printf '  would run: %s\n' "$*"; fi; }

NEEDS=()
pick_docker() { DOCKER=(docker); docker info >/dev/null 2>&1 || DOCKER=(sudo docker); }  # sudo only if you can't use Docker yourself

check_box() {
  echo "Checking this machine ($SCOPE install)"
  local fatal=0
  [[ $(uname -m) == aarch64 ]] && ok "arm64" || { bad "not arm64 ($(uname -m)); DGX-kit targets DGX Spark"; fatal=1; }

  if [[ -r /proc/driver/nvidia/version ]]; then
    ok "NVIDIA driver $(grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' /proc/driver/nvidia/version | head -1)"
  else
    bad "NVIDIA driver not loaded"; fatal=1
  fi
  # The GB10 may not list itself under /proc/driver/nvidia; ask the driver once as a fallback.
  local gpu_name=""
  gpu_name=$(grep -hs "^Model:" /proc/driver/nvidia/gpus/*/information 2>/dev/null | head -1 | sed 's/^Model:[[:space:]]*//')
  # The GB10 reports "Model: Unknown" there, so only trust a real name.
  [[ $gpu_name == Unknown* ]] && gpu_name=""
  [[ -n $gpu_name ]] || gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  if [[ $gpu_name == *GB10* ]]; then
    ok "GPU: $gpu_name"
  elif [[ -n $gpu_name ]]; then
    miss "GPU is $gpu_name, not a GB10; DGX-kit is built for the DGX Spark"
  else
    miss "GB10 not confirmed from /proc/driver/nvidia; continuing"
  fi

  . /etc/os-release 2>/dev/null && ok "OS: ${PRETTY_NAME:-unknown}"

  local free_gb
  free_gb=$(df -BG --output=avail "${HOME}" | tail -1 | tr -dc 0-9)
  (( free_gb >= 100 )) && ok "${free_gb} GB free in home" || miss "only ${free_gb} GB free; models need room"

  if curl -fsS -m 5 -o /dev/null https://huggingface.co 2>/dev/null; then ok "internet (huggingface.co)"; else miss "cannot reach huggingface.co"; fi

  if command -v docker >/dev/null; then ok "Docker $(docker --version | awk '{print $3}' | tr -d ,)"; else miss "Docker Engine"; NEEDS+=(docker); fi
  if command -v nvidia-ctk >/dev/null; then ok "NVIDIA Container Toolkit"; else miss "NVIDIA Container Toolkit"; NEEDS+=(toolkit); fi
  if id -nG "$USER" | grep -qw docker; then ok "$USER in docker group"; else miss "$USER in docker group"; NEEDS+=(group); fi
  if command -v docker >/dev/null && ! docker info >/dev/null 2>&1; then
    if [[ $SCOPE == user ]]; then bad "$USER can't use Docker without sudo yet (log out and in after joining the docker group)"; fatal=1
    else miss "Docker without sudo (the image is built with sudo, so this is fine here)"; fi
  fi
  if [[ $SCOPE == user ]] && (( ${#NEEDS[@]} )); then
    bad "a user install can't set up: ${NEEDS[*]}. Run without --user (needs sudo) or ask an administrator."
    fatal=1
  fi
  if "${CTL[@]}" is-active --quiet "$SERVICE" 2>/dev/null; then ok "DGX-kit already running (this run upgrades it)"; fi

  (( fatal == 0 )) || { echo "This machine can't run DGX-kit."; exit 1; }
}

port_used() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }  # something answers there

prev() {  # a value from the config of an earlier install of this service, if there is one and we may read it
  "${SUDO_N[@]}" sed -n "s/^$1=//p" "$CONF_DIR/config.env" 2>/dev/null | head -1 || true
}

ask_port() {  # ask_port VAR "question" default [own]; re-asks while the port is taken by something else. own: a port our running service already holds
  local var=$1 own=${4:-}
  set -- "$1" "$2" "$3"
  while :; do
    ask "$@"
    [[ -n $own && ${!var} == "$own" ]] && break  # our own previous install: it is stopped and restarted below
    port_used "${!var}" || break
    echo "  port ${!var} is already in use on this machine; pick another"
    # Without sudo this only names your own processes; a root or Docker owner shows as a listener with no name.
    ss -ltnpH "sport = :${!var}" 2>/dev/null | head -1 | sed 's/^/    listening: /'
    [[ -z ${!pre:-} ]] || { echo "  (it was set by DGXKIT_INSTALL_$var)"; exit 1; }
    set -- "$1" "$2" "$(( ${!var} + 1 ))"
  done
}

ask() {  # ask VAR "question" default; DGXKIT_INSTALL_<VAR> answers it ahead of time
  local var=$1 q=$2 def=$3 ans
  pre=DGXKIT_INSTALL_$var
  if [[ -n ${!pre:-} ]]; then printf -v "$var" '%s' "${!pre}"; return; fi
  read -r -p "  $q [$def]: " ans </dev/tty || true
  printf -v "$var" '%s' "${ans:-$def}"
}

ask_questions() {
  # One password prompt now, so an earlier install's answers can be read and the later steps don't stall.
  if [[ $MODE == install && $SCOPE == system ]]; then echo "This install needs administrator rights."; sudo -v; fi
  local p_models p_port p_bind p_gw own_port="" own_gw=""
  p_models=$(prev DGXKIT_MODELS_DIR); p_port=$(prev DGXKIT_PORT); p_bind=$(prev DGXKIT_BIND); p_gw=$(prev DGXKIT_GATEWAY_PORT)
  if "${CTL[@]}" is-active --quiet "$SERVICE" 2>/dev/null; then own_port=$p_port; fi
  if [[ -n $(docker ps -q --filter label=dgxkit.gateway 2>/dev/null) ]]; then own_gw=$p_gw; fi
  echo
  echo "A few questions (Enter keeps the default${p_port:+; defaults are your earlier answers})"
  ask MODELS_DIR "Where should models be stored?" "${p_models:-$HOME/models}"
  ask_port PORT "Dashboard port" "${p_port:-3000}" "$own_port"
  ask BIND "Reachable from the LAN (lan) or this machine only (local)?" "$([[ $p_bind == 127.0.0.1 ]] && echo local || echo lan)"
  ADMIN_PASSWORD=${DGXKIT_INSTALL_ADMIN_PASSWORD:-}
  KEEP_PASSWORD=no
  if [[ -z $ADMIN_PASSWORD ]] && "${SUDO_N[@]}" test -s "$STATE_DIR/admin.pw" 2>/dev/null; then
    # An earlier install has one: Enter, or no terminal to ask on, keeps it.
    read -r -s -p "  Admin password (Enter keeps the current one): " ADMIN_PASSWORD </dev/tty || true; echo
    [[ -n $ADMIN_PASSWORD ]] || KEEP_PASSWORD=yes
  fi
  while [[ $KEEP_PASSWORD != yes ]] && (( ${#ADMIN_PASSWORD} < 8 )); do
    if ! read -r -s -p "  Admin password for the dashboard (8+ characters): " ADMIN_PASSWORD </dev/tty; then
      [[ $MODE == dry ]] && { ADMIN_PASSWORD=dry-run-only; break; }
      echo; echo "  no terminal to ask on; set DGXKIT_INSTALL_ADMIN_PASSWORD (8+ characters)"; exit 1
    fi
    echo
  done
  HF_TOKEN=${DGXKIT_INSTALL_HF_TOKEN-}
  if [[ -z ${DGXKIT_INSTALL_HF_TOKEN+x} ]]; then
    read -r -s -p "  Hugging Face token (optional, Enter to skip): " HF_TOKEN </dev/tty || true; echo
  fi
  ask_port GATEWAY_PORT "Port for the LiteLLM gateway that publishes your models" "${p_gw:-4000}" "$own_gw"
  ask PULL_NOW "Pull engine images now? (yes/no)" "yes"
  UPGRADE=no
  [[ $SCOPE == system ]] && ask UPGRADE "Update all system packages first? Can change the kernel or NVIDIA driver (yes/no)" "no"
  READONLY=${DGXKIT_INSTALL_READONLY:-no}  # yes: watch only, DGX-kit won't start, stop or pull anything
}

install_missing() {
  [[ $SCOPE == system ]] || return 0
  echo
  echo "Installing what's missing"
  [[ $UPGRADE == yes ]] && run sudo apt-get update && run sudo apt-get -y upgrade
  for n in "${NEEDS[@]}"; do
    case $n in
      docker)  run sudo apt-get install -y docker.io ;;
      toolkit) run sudo apt-get install -y nvidia-container-toolkit
               run sudo nvidia-ctk runtime configure --runtime=docker
               run sudo systemctl restart docker ;;
      group)   run sudo usermod -aG docker "$USER" ;;
    esac
  done
}

write_config() {
  echo
  echo "Writing settings"
  local bind_addr=0.0.0.0; [[ $BIND == local ]] && bind_addr=127.0.0.1
  run "${SUDO[@]}" mkdir -p "$CONF_DIR" "$STATE_DIR"
  run mkdir -p "$MODELS_DIR"
  # One key for every client of the gateway. A reinstall keeps the old one so clients keep working.
  GATEWAY_KEY=$("${SUDO_N[@]}" sed -n 's/^LITELLM_MASTER_KEY=//p' "$CONF_DIR/config.env" 2>/dev/null || true)
  [[ -n $GATEWAY_KEY ]] || GATEWAY_KEY="sk-dgxkit-$(head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')"
  local body
  body=$(cat <<EOT
DGXKIT_MODELS_DIR=$MODELS_DIR
DGXKIT_PORT=$PORT
DGXKIT_BIND=$bind_addr
DGXKIT_GATEWAY_PORT=$GATEWAY_PORT
LITELLM_MASTER_KEY=$GATEWAY_KEY
DGXKIT_PULL_IMAGES=$PULL_NOW
DGXKIT_CACHE_DIR=$CACHE_DIR
DGXKIT_HOME=$HOME
HF_TOKEN=$HF_TOKEN
EOT
)
  [[ $READONLY == yes ]] && body+=$'\nDGXKIT_READONLY=1'
  if [[ $MODE == install ]]; then
    printf '%s\n' "$body" | "${SUDO[@]}" tee "$CONF_DIR/config.env" >/dev/null
    "${SUDO[@]}" chmod 600 "$CONF_DIR/config.env"
    # Store only a hash of the admin password.
    if [[ $KEEP_PASSWORD == yes ]]; then echo "  keeping the current admin password"; else
    printf '%s' "$ADMIN_PASSWORD" | python3 -c 'import hashlib,os,sys; s=os.urandom(16); print(s.hex()+":"+hashlib.scrypt(sys.stdin.buffer.read(),salt=s,n=2**14,r=8,p=1).hex())' \
      | "${SUDO[@]}" tee "$STATE_DIR/admin.pw" >/dev/null
    "${SUDO[@]}" chmod 600 "$STATE_DIR/admin.pw"
    fi
  else
    echo "  would write $CONF_DIR/config.env (mode 600) and a scrypt hash of the admin password"
  fi
}

build_image() {
  echo
  echo "Building the DGX-kit image (a re-run rebuilds it, which is how an update reaches the service)"
  [[ -f $ROOT_DIR/Dockerfile ]] || { bad "no Dockerfile in $ROOT_DIR; run this script from a DGX-kit checkout"; exit 1; }
  run mkdir -p "$CACHE_DIR"
  run "${DOCKER[@]}" build -t "$IMAGE" "$ROOT_DIR"
}

render_unit() {
  # A user service runs the container as you, so the files it makes in your folders are yours.
  local as_user="" after=""
  if [[ $SCOPE == user ]]; then
    as_user="--user $(id -u):$(id -g) --group-add $(getent group docker | cut -d: -f3) -e HOME=/tmp"
  else
    after=$'After=docker.service\nRequires=docker.service'
  fi
  cat <<EOT
[Unit]
Description=DGX-kit dashboard and model manager
$after

[Service]
EnvironmentFile=$CONF_DIR/config.env
ExecStartPre=-/usr/bin/docker rm -f $SERVICE
ExecStart=/usr/bin/docker run --name $SERVICE --gpus all --network host --pid host $as_user \\
  --env-file $CONF_DIR/config.env \\
  -v /var/run/docker.sock:/var/run/docker.sock \\
  -v /sys:/host/sys:ro -v /proc:/host/proc:ro -e DGXKIT_ROOT=/host \\
  -v /home:/home:ro -v $STATE_DIR:$STATE_DIR -v \${DGXKIT_MODELS_DIR}:\${DGXKIT_MODELS_DIR} \\
  -v $CACHE_DIR:$CACHE_DIR \\
  -e DGXKIT_STATE_DIR=$STATE_DIR \\
  $IMAGE
ExecStop=/usr/bin/docker stop $SERVICE
Restart=always

[Install]
WantedBy=$WANTED_BY
EOT
}

install_service() {
  echo
  echo "Starting DGX-kit"
  local unit
  unit=$(render_unit)
  if [[ $MODE == install ]]; then
    "${SUDO[@]}" mkdir -p "$UNIT_DIR"
    printf '%s\n' "$unit" | "${SUDO[@]}" tee "$UNIT_DIR/$SERVICE.service" >/dev/null
    "${SUDO[@]}" "${CTL[@]}" daemon-reload
    "${SUDO[@]}" "${CTL[@]}" enable "$SERVICE"
    "${SUDO[@]}" "${CTL[@]}" restart "$SERVICE"  # not "enable --now": that leaves an already-running service on the old image
    if [[ $SCOPE == user ]]; then
      loginctl enable-linger "$USER" 2>/dev/null || echo "  note: couldn't enable lingering; the service stops when you log out until an administrator runs: sudo loginctl enable-linger $USER"
    fi
  else
    echo "  would write $UNIT_DIR/$SERVICE.service and enable it:"
    printf '%s\n' "$unit" | sed 's/^/    | /'
  fi
  install_helpers
  local host; host=$(hostname -I 2>/dev/null | awk '{print $1}')
  [[ $BIND == local ]] && host=127.0.0.1
  echo
  echo "DGX-kit: http://${host:-localhost}:$PORT"
  echo "Models, once started, are published at http://${host:-localhost}:$GATEWAY_PORT/v1"
  if [[ $MODE == install && $ACTION == install ]]; then echo "Gateway key (also in $CONF_DIR/config.env): $GATEWAY_KEY"; fi
}

restart_service() {
  if [[ $MODE != install ]]; then echo "  would restart $SERVICE"; return; fi
  if "${SUDO_N[@]}" true 2>/dev/null; then
    "${SUDO[@]}" "${CTL[@]}" restart "$SERVICE"; return
  fi
  # No sudo without a password: stop the container and let the service (Restart=always) bring it back on the new image.
  local before; before=$("${DOCKER[@]}" inspect "$SERVICE" --format '{{.State.StartedAt}}' 2>/dev/null || true)
  "${DOCKER[@]}" stop "$SERVICE" >/dev/null 2>&1 || true
  local i now
  for i in $(seq 1 40); do
    now=$("${DOCKER[@]}" inspect "$SERVICE" --format '{{.State.Status}} {{.State.StartedAt}}' 2>/dev/null || true)
    [[ $now == running\ * && $now != "running $before" ]] && return 0
    sleep 2
  done
  bad "the service didn't come back by itself; run: sudo systemctl restart $SERVICE"; exit 1
}

apply_update() {
  local want have
  want=$(render_unit); have=$(cat "$UNIT_DIR/$SERVICE.service" 2>/dev/null || true)
  if [[ $want != "$have" ]]; then
    echo "The service file changed in this version; rewriting it (this one needs administrator rights)"
    if [[ $MODE == install ]]; then
      [[ $SCOPE == system ]] && sudo -v
      "${SUDO[@]}" mkdir -p "$UNIT_DIR"
      printf '%s\n' "$want" | "${SUDO[@]}" tee "$UNIT_DIR/$SERVICE.service" >/dev/null
      "${SUDO[@]}" "${CTL[@]}" daemon-reload
      "${SUDO[@]}" "${CTL[@]}" enable "$SERVICE"
    else
      echo "  would rewrite $UNIT_DIR/$SERVICE.service:"; printf '%s\n' "$want" | sed 's/^/    | /'
    fi
  else
    ok "service file unchanged"
  fi
  restart_service
}

report_version() {
  [[ $MODE == install ]] || return 0
  local v="" i
  for i in $(seq 1 20); do
    v=$("${DOCKER[@]}" exec "$SERVICE" python -c 'from dgxkit.app import app_version; print(app_version())' 2>/dev/null) && break
    sleep 2
  done
  echo
  echo "Updated${v:+ to version $v}. To roll back: ${DOCKER[*]} tag ${IMAGE%%:*}:previous $IMAGE && ${DOCKER[*]} stop $SERVICE"
}

install_helpers() {  # the command and the installer copy it runs: both in your home folder, so no sudo
  local bin=$HOME/.local/bin/$SERVICE share=$HOME/.local/share/$SERVICE-installer scopeflag=""
  [[ $SCOPE == user ]] && scopeflag="--user"
  if [[ $MODE != install ]]; then echo "  would install the '$SERVICE' command in $bin"; return; fi
  mkdir -p "$(dirname "$bin")" "$share"
  cp "$ROOT_DIR/installer/install.sh" "$share/install.sh"
  cat > "$bin" <<'TEMPLATE'
#!/usr/bin/env bash
# @SERVICE@: update or check the DGX-kit install on this machine. Written by installer/install.sh.
set -euo pipefail
SERVICE=@SERVICE@
IMAGE=@IMAGE@
REPO=${DGXKIT_UPDATE_REPO:-@REPO@}
INSTALLER="$HOME/.local/share/@SERVICE@-installer/install.sh"
run_installer() { DGXKIT_INSTALL_SERVICE=$SERVICE DGXKIT_IMAGE=$IMAGE exec bash "$INSTALLER" @SCOPEFLAG@ "$@"; }
usage() {
  cat <<'U'
usage: @SERVICE@ update [PACKAGE.tar.gz] [--dry-run]   update to the latest from git, or from a package file
       @SERVICE@ version                               the version that is running
U
}
case "${1:-help}" in
  update)
    shift
    if [[ ${1:-} == *.tar.gz ]]; then f=$1; shift; run_installer --updatepath "$f" "$@"
    else run_installer --updaterepo "$REPO" "$@"; fi ;;
  version) docker exec "$SERVICE" python -c 'from dgxkit.app import app_version; print(app_version())' 2>/dev/null \
             || sudo docker exec "$SERVICE" python -c 'from dgxkit.app import app_version; print(app_version())' ;;
  help|-h|--help) usage ;;
  *) usage >&2; exit 2 ;;
esac
TEMPLATE
  sed -i.bak "s|@SERVICE@|$SERVICE|g; s|@IMAGE@|$IMAGE|g; s|@REPO@|$DEFAULT_UPDATE_REPO|g; s|@SCOPEFLAG@|$scopeflag|g" "$bin" && rm -f "$bin.bak"
  chmod +x "$bin"
  echo "  command installed: $SERVICE update   (and: $SERVICE version)"
  case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) echo "  note: add $HOME/.local/bin to your PATH to run '$SERVICE' by name" ;; esac
}

add_config_key() {  # a setting newer versions need, appended to the config of an older install if it isn't there
  local key=$1 value=$2
  [[ -z $(prev "$key") ]] || return 0
  echo "  adding $key to $CONF_DIR/config.env"
  if [[ $MODE == install ]]; then printf '%s=%s\n' "$key" "$value" | "${SUDO[@]}" tee -a "$CONF_DIR/config.env" >/dev/null; fi
}

update_from_here() {
  echo "Updating the $SCOPE install called $SERVICE from $ROOT_DIR"
  [[ -f $ROOT_DIR/Dockerfile ]] || { bad "no Dockerfile in $ROOT_DIR; this isn't a DGX-kit package"; exit 1; }
  if [[ ! -f $UNIT_DIR/$SERVICE.service ]]; then
    bad "no install called $SERVICE found ($UNIT_DIR/$SERVICE.service is missing). Nothing changed."
    echo "       To install for the first time, run this script without --update."; exit 1
  fi
  # Settings newer versions need are added only when they can be written without asking for a password.
  if "${SUDO_N[@]}" test -r "$CONF_DIR/config.env" 2>/dev/null; then
    add_config_key DGXKIT_HOME "$HOME"
    add_config_key DGXKIT_CACHE_DIR "$CACHE_DIR"
  fi
  echo
  echo "Keeping the current image as ${IMAGE%%:*}:previous"
  run "${DOCKER[@]}" tag "$IMAGE" "${IMAGE%%:*}:previous" 2>/dev/null || echo "  (no earlier image to keep)"
  build_image
  apply_update
  install_helpers
  report_version
}

update() {
  local flags=(--update) top
  [[ $SCOPE == user ]] && flags+=(--user)
  [[ $MODE == dry ]] && flags+=(--dry-run)
  UNPACKED=$(mktemp -d "${TMPDIR:-/tmp}/dgxkit-update.XXXXXX")  # global, so the exit trap can still see it
  trap 'rm -rf "$UNPACKED"' EXIT
  if [[ -n $UPDATE_REPO ]]; then
    command -v git >/dev/null || { bad "git isn't installed here"; exit 1; }
    echo "Fetching the latest from $UPDATE_REPO"
    if ! git clone -q --depth 1 "$UPDATE_REPO" "$UNPACKED/src" 2>"$UNPACKED/git.err"; then
      bad "couldn't fetch $UPDATE_REPO"; sed 's/^/       /' "$UNPACKED/git.err" | head -4
      echo "       Is it reachable from this machine, and does this machine have access to it (a deploy key or a token)?"
      echo "       Or update from a package file:  $SERVICE update PACKAGE.tar.gz"; exit 1
    fi
    top=src
    [[ -f $UNPACKED/src/installer/install.sh && -f $UNPACKED/src/Dockerfile ]] || { bad "$UPDATE_REPO doesn't look like a DGX-kit repository"; exit 1; }
    ok "fetched $(git -C "$UNPACKED/src" log -1 --format='%h %s' | cut -c1-70)"
  else
    [[ -f $UPDATE_FILE ]] || { bad "no such file: $UPDATE_FILE"; exit 1; }
    if [[ -f $UPDATE_FILE.sha256 ]]; then
      local want got; want=$(awk '{print $1}' "$UPDATE_FILE.sha256")
      got=$( (sha256sum "$UPDATE_FILE" 2>/dev/null || shasum -a 256 "$UPDATE_FILE") | awk '{print $1}')
      [[ $want == "$got" ]] || { bad "the checksum doesn't match $UPDATE_FILE.sha256; not using this file"; exit 1; }
      ok "checksum matches"
    fi
    local names; names=$(tar tzf "$UPDATE_FILE" 2>/dev/null) || { bad "$UPDATE_FILE isn't a .tar.gz"; exit 1; }
    if grep -qE '(^|/)\.\.(/|$)|^/' <<<"$names"; then bad "the package has paths outside its own folder; not unpacking it"; exit 1; fi
    names=$(sed 's#^\./##; s#/$##' <<<"$names")
    top=$(grep -E '^[^/]+/installer/install\.sh$' <<<"$names" | head -1 | cut -d/ -f1)
    if [[ -z $top ]] || ! grep -qx "$top/Dockerfile" <<<"$names"; then
      bad "$UPDATE_FILE doesn't look like a DGX-kit package (no installer/install.sh and Dockerfile in one top folder)"; exit 1
    fi
    tar xzf "$UPDATE_FILE" -C "$UNPACKED"
    ok "unpacked $top"
  fi
  # The update runs from the new installer, so the newest install logic is always the one that applies it.
  bash "$UNPACKED/$top/installer/install.sh" "${flags[@]}"
}

uninstall() {
  echo "Removing the $SCOPE install called $SERVICE"
  local DOCKER=(docker)
  docker info >/dev/null 2>&1 || DOCKER=(sudo docker)
  echo
  ask REMOVE_MODELS "Also stop and remove the model containers and the LiteLLM gateway that DGX-kit started? (yes/no)" "no"
  ask REMOVE_DATA "Also delete its saved settings, keys and recipes, and LiteLLM's database? (yes/no)" "no"
  echo
  echo "This will remove the service and the dashboard container and image."
  [[ $REMOVE_MODELS == yes ]] && echo "It will also remove every model container DGX-kit started (running models stop) and the LiteLLM gateway."
  [[ $REMOVE_DATA == yes ]] && echo "It will also delete $CONF_DIR and $STATE_DIR and LiteLLM's database volume. There is no undo."
  echo "It will not touch your models or the compiled-kernel caches in $CACHE_DIR."
  if [[ $MODE == install && ${DGXKIT_INSTALL_YES:-no} != yes ]]; then
    local reply=""
    read -r -p "  Type 'uninstall' to go ahead: " reply </dev/tty || { echo; echo "  no terminal to ask on; set DGXKIT_INSTALL_YES=yes to go ahead"; exit 1; }
    [[ $reply == uninstall ]] || { echo "  Nothing removed."; exit 1; }
  fi
  echo
  run "${SUDO[@]}" "${CTL[@]}" disable --now "$SERVICE" || true
  run "${SUDO[@]}" rm -f "$UNIT_DIR/$SERVICE.service"
  run "${SUDO[@]}" "${CTL[@]}" daemon-reload
  run "${SUDO[@]}" "${CTL[@]}" reset-failed "$SERVICE" || true  # stopping the container leaves the unit listed as failed
  run "${DOCKER[@]}" rm -f "$SERVICE" || true
  if [[ $REMOVE_MODELS == yes ]]; then
    local ids
    ids=$(for l in dgxkit.model dgxkit.gateway dgxkit.gateway-db; do "${DOCKER[@]}" ps -aq --filter "label=$l" 2>/dev/null; done)  # one query per label: several --filter label= are ANDed
    [[ -z $ids ]] || run "${DOCKER[@]}" rm -f $ids
  fi
  if [[ $REMOVE_DATA == yes ]]; then
    run "${DOCKER[@]}" volume rm dgxkit-gateway-pg || echo "  note: the LiteLLM database volume is still in use or already gone"
    run "${SUDO[@]}" rm -rf "$CONF_DIR" "$STATE_DIR"
  fi
  run "${DOCKER[@]}" rmi "$IMAGE" || true
  run rm -f "$HOME/.local/bin/$SERVICE"                       # the update command and the installer copy it runs
  run rm -rf "$HOME/.local/share/$SERVICE-installer"
  echo
  echo "Done."
  [[ $REMOVE_MODELS == yes ]] || echo "Model containers and the LiteLLM gateway were left running."
  [[ $REMOVE_DATA == yes ]] || echo "Settings and saved data are still in $CONF_DIR and $STATE_DIR."
}

if [[ $ACTION == uninstall ]]; then
  uninstall
  exit 0
fi
if [[ $ACTION == update ]]; then
  [[ $MODE == check ]] && MODE=dry
  pick_docker
  if [[ -n $UPDATE_FILE || -n $UPDATE_REPO ]]; then update; else update_from_here; fi
  exit 0
fi

check_box
if [[ $MODE == check ]]; then
  echo
  (( ${#NEEDS[@]} )) && echo "Would install: ${NEEDS[*]}" || echo "Nothing to install."
  exit 0
fi
ask_questions
install_missing
write_config
pick_docker
build_image
install_service
