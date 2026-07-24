#!/usr/bin/env bash
# Camon add-on entrypoint.
#
# The add-on is configured EXACTLY like a native install: a camon.toml file in
# the add-on's config folder (/addon_configs/..._camon, mounted here at
# /config). There is no options UI — run.sh just points Camon at that file and
# forces three values that are non-negotiable inside the container:
#
#   * update.enabled = false — in-container self-update is wrong for an add-on.
#     The container filesystem is ephemeral and updates flow through Home
#     Assistant's add-on store, so Camon's GitHub self-updater is disabled.
#   * http.port = 22666 — the add-on is reached exclusively through ingress,
#     which is wired to this internal port (ingress_port in config.yaml).
#   * storage.data_dir = /data/storage — /data is the add-on's persistent
#     volume, so recordings survive restarts/updates.
#
# These are applied with `--set` at startup, overriding whatever the file says.
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

echo "[camon-addon] using $CONFIG (update.enabled/http.port/storage.data_dir forced)"

# exec so Camon becomes PID 1 and receives SIGTERM directly for graceful stop.
exec camon --config "$CONFIG" \
  --set update.enabled=false \
  --set http.port=22666 \
  --set storage.data_dir=/data/storage
