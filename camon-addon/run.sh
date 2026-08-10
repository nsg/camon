#!/usr/bin/env bash
# Camon add-on entrypoint.
#
# The add-on is configured EXACTLY like a native install: a camon.toml file in
# the add-on's config folder (/addon_configs/..._camon, mounted here at
# /config). There is no options UI — run.sh just points Camon at that file and
# forces five values that are non-negotiable inside the container:
#
#   * update.enabled = false — in-container self-update is wrong for an add-on.
#     The container filesystem is ephemeral and updates flow through Home
#     Assistant's add-on store, so Camon's GitHub self-updater is disabled.
#   * http.port = 22666 — the add-on is reached exclusively through ingress,
#     which the catalog manifest wires to this internal port.
#   * http.bind = 0.0.0.0 — ingress reaches the container over the container
#     network, so binding loopback (or anything else) silently breaks it.
#   * storage.data_dir = /data/storage — /data is the add-on's persistent
#     volume, so recordings survive restarts/updates.
#   * http.allow_open = true — ingress IS the authentication layer here (Home
#     Assistant authenticates the user before proxying), and the internal port
#     is only reachable from the container network. Without this, Camon would
#     see a 0.0.0.0 bind with no [http] token and generate a token of its own
#     to guard its write endpoints — a secret ingress could never present, so
#     the add-on's motion settings and mask editor would stop saving.
#
# These are applied with `--set` at startup, overriding whatever the file says.
#
# Additionally, if a Supervisor-managed MQTT broker is installed, the external
# catalog manifest's `mqtt:want` service grant lets us apply its connection
# details with --set, overriding any [mqtt] section in camon.toml. This is
# conditional — camon runs fine without it if no broker is installed.
set -euo pipefail

CONFIG=/config/camon.toml
EXAMPLE_SRC=/usr/local/share/camon/config.toml.example
EXAMPLE_DST=/config/camon.toml.example

mkdir -p /data/storage

# Seed a commented example into the add-on config folder on first start, so the
# user has a fully documented template to copy from.
if [ ! -f "$EXAMPLE_DST" ]; then
  cp "$EXAMPLE_SRC" "$EXAMPLE_DST"
  echo "[camon-addon] seeded $EXAMPLE_DST"
fi

if [ ! -f "$CONFIG" ]; then
  cat >&2 <<'EOF'
[camon-addon] No config file found at /config/camon.toml.

Camon is configured by a camon.toml file — the SAME format as a native install
(nothing is set from the add-on's Configuration tab). Create it in the add-on's
config folder:

    /addon_configs/<repo>_camon/camon.toml

To create/edit files there you need a file-access add-on — install one of:
File editor, Studio Code Server, SSH, or Samba. A fully commented template has
already been placed next to it as camon.toml.example: copy or rename that to
camon.toml and edit it (at minimum, add your [[cameras]]).

Then restart this add-on.
EOF
  exit 1
fi

echo "[camon-addon] using $CONFIG (update.enabled/http.port/http.bind/http.allow_open/storage.data_dir forced)"

# --- MQTT auto-configuration from a Supervisor-managed broker --------------
# The catalog manifest requests `services: - mqtt:want`, which grants access
# to the Supervisor's /services/mqtt endpoint if an MQTT broker add-on (e.g.
# Mosquitto) is installed. When it responds, we force mqtt.enabled/host/port/
# username/password with --set — these take precedence over any [mqtt] values
# in camon.toml itself. When no broker is installed the call fails and we start
# camon as-is: whatever (if anything) camon.toml's [mqtt] section says is used
# unmodified.
MQTT_ARGS=()
if MQTT_JSON=$(curl -sf -H "Authorization: Bearer ${SUPERVISOR_TOKEN}" http://supervisor/services/mqtt) \
    && [ "$(printf '%s' "$MQTT_JSON" | jq -r '.result')" = "ok" ]; then
  MQTT_HOST=$(printf '%s' "$MQTT_JSON" | jq -r '.data.host')
  MQTT_PORT=$(printf '%s' "$MQTT_JSON" | jq -r '.data.port')
  MQTT_USERNAME=$(printf '%s' "$MQTT_JSON" | jq -r '.data.username // empty')
  MQTT_PASSWORD=$(printf '%s' "$MQTT_JSON" | jq -r '.data.password // empty')
  MQTT_ARGS+=(--set mqtt.enabled=true --set "mqtt.host=${MQTT_HOST}" --set "mqtt.port=${MQTT_PORT}")
  [ -n "$MQTT_USERNAME" ] && MQTT_ARGS+=(--set "mqtt.username=${MQTT_USERNAME}")
  [ -n "$MQTT_PASSWORD" ] && MQTT_ARGS+=(--set "mqtt.password=${MQTT_PASSWORD}")
  echo "[camon-addon] MQTT broker found via Supervisor (host=${MQTT_HOST} port=${MQTT_PORT}); mqtt.* forced (password not logged)"
else
  echo "[camon-addon] no Supervisor MQTT service found; camon.toml's [mqtt] section (if any) is used as-is"
fi

# exec so Camon becomes PID 1 and receives SIGTERM directly for graceful stop.
exec camon --config "$CONFIG" \
  --set update.enabled=false \
  --set http.port=22666 \
  --set http.bind=0.0.0.0 \
  --set http.allow_open=true \
  --set storage.data_dir=/data/storage \
  "${MQTT_ARGS[@]}"
