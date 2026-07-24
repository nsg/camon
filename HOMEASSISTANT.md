# Camon as a Home Assistant add-on

This repository doubles as a [Home Assistant add-on repository](https://developers.home-assistant.io/docs/add-ons/repository).
Home Assistant is not the primary way Camon is meant to run — it is a native
Linux service first (see the main [README](README.md)) — but if you run Home
Assistant OS or Supervised, you can install Camon as an add-on and get its web
UI embedded directly in the Home Assistant sidebar via **ingress**.

> **amd64 only.** The add-on builds and runs on `amd64` (x86_64) only. ARM
> (`aarch64`) is deliberately deferred and not offered in `config.yaml`'s
> `arch` list. Camon spawns `ffmpeg`/`ffprobe` and is only tested on x86_64.

## Install

1. In Home Assistant, go to **Settings → Add-ons → Add-on store**.
2. Open the **⋮** menu (top-right) → **Repositories**, paste
   `https://github.com/nsg/camon`, and click **Add**. (The button in the
   README does this for you.)
3. The **Camon** add-on appears in the store. Open it and click **Install**.
   The image is built from source on first install (Rust compile + ffmpeg), so
   the first build takes a few minutes.
4. Configure the options (see below), then **Start**. With ingress enabled,
   **Camon** shows up in the sidebar — click it to open the web UI.

## Repository layout

Standard HA add-on repository conventions:

```
repository.yaml        # repository manifest (name/url/maintainer) at repo root
camon-addon/           # one add-on == one top-level folder
  config.yaml          # add-on manifest (slug, arch, ingress, options, schema)
  Dockerfile           # multi-stage: build Camon from source, run on Debian+ffmpeg
  run.sh               # options.json -> Camon JSON config, then exec camon
```

The Home Assistant supervisor builds the image with `camon-addon/` as the
Docker context, so the Rust source is not in the context — the build stage
instead clones this repository at the version tag matching `config.yaml`
(`v0.3.0`), so the add-on always compiles the exact released source. (That tag
must exist when the image is built — tag the release before installing.)

## Ingress and the sidebar panel

`config.yaml` wires ingress and the sidebar panel:

```yaml
ingress: true
ingress_port: 8080      # Camon's internal HTTP port
panel_icon: mdi:cctv
panel_title: Camon
panel_admin: true
```

Home Assistant proxies the ingress URL to the add-on's `ingress_port` and
**strips the ingress path prefix** before the request reaches Camon. That means
the Camon web UI must work behind an arbitrary base path. It does: every asset
href, API `fetch`, HLS playlist/segment URL, and image `src` in the UI is a
**relative** URL (no leading `/`), and the app uses hash routing (`#/camera/x`)
which leaves the document path — and therefore the relative-URL base —
untouched. So the UI resolves correctly whether it's served at `/` (systemd
install) or under `…/api/hassio_ingress/<token>/` (add-on).

## Options mapping

The add-on exposes a pragmatic subset of Camon's full config. `run.sh`
translates `/data/options.json` into the nested JSON config Camon reads (Camon
now loads JSON as well as TOML — see below), then runs `camon --config`.

| Add-on option     | Type (HA schema) | Maps to Camon config field                          |
| ----------------- | ---------------- | --------------------------------------------------- |
| `analytics`       | `bool`           | `analytics.enabled`                                 |
| `object_detection`| `bool`           | `analytics.object_detection.enabled`                |
| `ollama_url`      | `url`            | `analytics.object_detection.ollama.url`             |
| `ollama_model`    | `str`            | `analytics.object_detection.ollama.model`           |
| `storage`         | `bool`           | `storage.enabled`                                   |
| `stathost_url`    | `url?` (optional)| `storage.stathost.url` (section emitted only if set)|
| `stathost_bucket` | `str?` (optional)| `storage.stathost.bucket` (defaults to `camon`)     |
| `stathost_token`  | `password?`      | `storage.stathost.token`                            |
| `cameras`         | list of `{id, url}` | `cameras` (passthrough)                          |

Anything not exposed (retention days, padding, motion tuning, confidence
threshold, ollama fallback/timeout, …) uses Camon's built-in defaults. To tune
those, edit the generated config or run Camon outside Home Assistant.

Two values are **forced by `run.sh`** and cannot be overridden from options:

- **`update.enabled = false`.** Camon's built-in GitHub self-updater is
  disabled inside the add-on. The container filesystem is ephemeral and updates
  must flow through the Home Assistant add-on store, so an in-container binary
  swap would be pointless (lost on rebuild) and would fight Home Assistant's
  own update mechanism.
- **`storage.data_dir = /data/storage`.** `/data` is the add-on's own
  persistent volume — always mounted and preserved across restarts/updates, no
  `map:` entry required — so recordings survive there automatically.
- **`http.port = 8080`.** The add-on is reached exclusively through ingress,
  which is wired to internal port 8080; the port is pinned so that wiring can't
  be broken from options.

## JSON config support (used by the add-on)

Camon loads its config from **TOML or JSON** with the identical schema; the
parser is chosen by file extension (`.json` → JSON, otherwise TOML). You can
also point Camon at any path with `--config`:

```
camon --config /data/camon.json
```

With no `--config`, Camon looks for `./config.toml` first, then `./config.json`.
The add-on relies on this: `run.sh` writes `/data/camon.json` from the add-on
options and starts `camon --config /data/camon.json`. See
[`config.toml.example`](config.toml.example) for the full field reference (the
JSON form is the same structure, just JSON-encoded).

## Building the image

The `Dockerfile` is multi-stage:

1. **Build** — `rust:1-bookworm`, `cargo build --release --locked`.
2. **Runtime** — `debian:bookworm-slim` with `ffmpeg`, `ca-certificates`, and
   `jq` installed; the compiled binary and `run.sh` are copied in.

It intentionally does **not** use a Home Assistant base image: those are
Alpine/musl, and a glibc binary from the Rust toolchain image won't run on
musl. Keeping both stages on Debian bookworm avoids a musl cross-compile.
`config.yaml` sets `init: false`, so the `CMD ["/run.sh"]` runs as PID 1 and
Camon receives `SIGTERM` directly for a graceful shutdown.

> The Dockerfile and `config.yaml` here were validated by conformance to the
> Home Assistant add-on docs; they were **not** built/run in CI in this
> environment (no Docker available), so treat the first `ha` build as the real
> smoke test.
