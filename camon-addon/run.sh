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

mkdir -p /data/storage

# Build the Camon config from the add-on options. Optional stathost fields are
# only emitted when a URL was provided (empty/absent => local-disk storage).
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

echo "[camon-addon] wrote $CONFIG (update.enabled forced false, data_dir=/data/storage)"

# exec so Camon becomes PID 1 and receives SIGTERM directly for graceful stop.
exec camon --config "$CONFIG"
