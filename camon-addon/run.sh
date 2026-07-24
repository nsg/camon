#!/usr/bin/env bash
# Camon add-on entrypoint.
#
# Home Assistant writes the user's options to /data/options.json (matching the
# `schema` in config.yaml). We translate that flat option set into the nested
# JSON config Camon reads (same schema as config.toml), then hand it over with
# `--config`. Two values are FORCED here regardless of what the user set:
#
#   * update.enabled = false — in-container self-update is wrong for an add-on.
#     The container filesystem is ephemeral and updates flow through Home
#     Assistant's add-on store, so Camon's GitHub self-updater is disabled.
#   * storage.data_dir = /data/storage — /data is the add-on's persistent
#     volume, so recordings survive restarts/updates.
set -euo pipefail

OPTIONS=/data/options.json
CONFIG=/data/camon.json
# Full-config override: the user may place a camon.json with the EXACT same
# structure as the project's config.toml (JSON-encoded — Camon reads both
# formats natively) in the add-on's config folder (/addon_configs/..._camon,
# mounted here at /config). When present it replaces the options-derived
# config entirely; only the container-forced keys below are merged on top.
USER_CONFIG=/config/camon.json

mkdir -p /data/storage

if [ -f "$USER_CONFIG" ]; then
  # Deep-merge the forced keys over the user's full config.
  jq -s '.[0] * {
    update: { enabled: false },
    http: { port: 22666 },
    storage: { data_dir: "/data/storage" }
  }' "$USER_CONFIG" > "$CONFIG"
  echo "[camon-addon] using full config from $USER_CONFIG" \
       "(update.enabled/http.port/storage.data_dir forced)"
else
  # Build the Camon config from the add-on options. Optional stathost fields
  # are only emitted when a URL was provided (empty/absent => local disk).
  jq '{
    update: { enabled: false },
    http: { port: 22666 },
    analytics: {
      enabled: .analytics,
      object_detection: {
        enabled: .object_detection,
        ollama: { url: .ollama_url, model: .ollama_model }
      }
    },
    storage: (
      { enabled: .storage, data_dir: "/data/storage" }
      + ( if ((.stathost_url // "") != "") then
            { stathost: {
                url: .stathost_url,
                bucket: (.stathost_bucket // "camon"),
                token: (.stathost_token // "")
            } }
          else {} end )
    ),
    cameras: (.cameras // [])
  }' "$OPTIONS" > "$CONFIG"
  echo "[camon-addon] wrote $CONFIG from add-on options" \
       "(update.enabled forced false, data_dir=/data/storage)"
fi

# exec so Camon becomes PID 1 and receives SIGTERM directly for graceful stop.
exec camon --config "$CONFIG"
