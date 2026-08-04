# Camon as a Home Assistant add-on

> **This file documents the Home Assistant add-on only.** (It is named
> `DOCS.md` because Home Assistant requires that exact filename for the add-on
> Documentation tab.) If you are not running Home Assistant, this file is not
> for you — see the main [README](https://github.com/nsg/camon#readme) for the
> native installation.

This repository doubles as a [Home Assistant add-on repository](https://developers.home-assistant.io/docs/add-ons/repository).
Home Assistant is not the primary way Camon is meant to run — it is a native
Linux service first — but if you run Home Assistant OS or Supervised, you can
install Camon as an add-on and get its web UI embedded directly in the Home
Assistant sidebar via **ingress**.

> **amd64 only.** The add-on builds and runs on `amd64` (x86_64) only. ARM
> (`aarch64`) is deliberately deferred and not offered in `config.yaml`'s
> `arch` list. Camon spawns `ffmpeg`/`ffprobe` and is only tested on x86_64.

## Install

1. In Home Assistant, go to **Settings → Add-ons → Add-on store**.
2. Open the **⋮** menu (top-right) → **Repositories**, paste
   `https://github.com/nsg/camon`, and click **Add**. (The button in the
   README does this for you.)
3. The **Camon** add-on appears in the store. Open it and click **Install**.
   This pulls a prebuilt image from GHCR (`ghcr.io/nsg/amd64-addon-camon`) —
   nothing is compiled on your machine.
4. Create the configuration file (see [Configuration](#configuration) below),
   then **Start**. With ingress enabled, **Camon** shows up in the sidebar —
   click it to open the web UI.
5. Optional but recommended: turn on **Watchdog** on the add-on's page. Camon
   exits when it loses a task it cannot run without — a dead recorder is meant
   to become a restart rather than a process that looks healthy and records
   nothing — and the Supervisor only restarts a stopped add-on when its
   Watchdog is on. It is off by default.

## Repository layout

Standard HA add-on repository conventions:

```
repository.yaml        # repository manifest (name/url/maintainer) at repo root
camon-addon/           # one add-on == one top-level folder
  config.yaml          # add-on manifest (slug, arch, image, ingress, map)
  Dockerfile           # multi-stage: build Camon from source, run on Debian+ffmpeg
  run.sh               # points Camon at /config/camon.toml (forcing 3 values), then exec camon
```

`config.yaml` sets `image: ghcr.io/nsg/{arch}-addon-camon`, so the Home
Assistant supervisor **pulls** the prebuilt image at the tag matching the
add-on version — it never builds the Dockerfile itself. The image is built and
pushed by GitHub Actions (`.github/workflows/addon.yml`); see
[Building the image](#building-the-image).

## Ingress and the sidebar panel

`config.yaml` wires ingress and the sidebar panel:

```yaml
ingress: true
ingress_port: 22666     # Camon's internal HTTP port ("camon" on a phone keypad)
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

## Configuration

The add-on has **no options UI**. It is configured by a `camon.toml` file with
the **exact same format as a native install**, so everything Camon supports is
available — see [`config.toml.example`](https://github.com/nsg/camon/blob/master/config.toml.example) for the full,
commented field reference.

Create the file in the add-on's config folder:

```
/addon_configs/<repo>_camon/camon.toml
```

That folder is the add-on's own config directory, mounted read-write at
`/config` inside the container via `map: addon_config:rw`. Home Assistant does
not give you a file browser for it out of the box, so you need a file-access
add-on to create and edit the file — any of **File editor**, **Studio Code
Server**, **SSH**, or **Samba** works.

On first start the add-on **seeds a fully commented `camon.toml.example`** into
that folder (copied from the project's `config.toml.example`). Copy or rename
it to `camon.toml`, add at least one `[[cameras]]` block, and restart the
add-on. Until `camon.toml` exists the add-on logs these instructions and exits.

### Values forced at startup

Five values are **forced by `run.sh`** via `camon --set`, overriding whatever
`camon.toml` says — they are non-negotiable inside the container:

- **`update.enabled = false`.** Camon's built-in GitHub self-updater is
  disabled inside the add-on. The container filesystem is ephemeral and updates
  must flow through the Home Assistant add-on store, so an in-container binary
  swap would be pointless (lost on rebuild) and would fight Home Assistant's
  own update mechanism. Camon defaults to this too, but the add-on pins it so
  the key cannot be turned on from `camon.toml`.
- **`http.port = 22666`.** The add-on is reached exclusively through ingress,
  which is wired to internal port 22666 — an uncommon port picked so it can
  never collide with another service even if host networking is ever enabled
  (with ingress-only access, no host port is bound at all). The port is pinned
  so that wiring can't be broken from the config file.
- **`http.bind = 0.0.0.0`.** Ingress reaches the add-on over the container
  network, so the listener has to accept connections from it. A `bind` of
  `127.0.0.1` in `camon.toml` would silently break ingress, hence the pin.
- **`http.allow_open = true`.** Standalone camon warns loudly at startup when
  its API is reachable over the network without an `[http] token`. In the
  add-on that warning does not apply: nothing outside the container can reach
  port 22666, and Home Assistant authenticates every user before proxying them
  through ingress — HA *is* the authentication layer here. Setting an `[http]
  token` as well is supported and harmless, but not needed.
- **`storage.data_dir = /data/storage`.** `/data` is the add-on's own
  persistent volume — always mounted and preserved across restarts/updates — so
  recordings survive there automatically. Camon marks the directory at startup
  with a small `.camon-volume` file and re-checks it once a minute, so a storage
  volume that goes away under a running camon is reported instead of quietly
  redirecting recordings elsewhere; on the add-on that check simply never fires.

Everything else — analytics, object detection, the Ollama server (including the
fallback), retention, motion tuning, remote stathost storage, and the camera
list — is set in `camon.toml` exactly as documented for a native install.

## MQTT / Home Assistant entities

Camon can publish [MQTT discovery](https://www.home-assistant.io/integrations/mqtt/#mqtt-discovery)
messages so each camera shows up as native Home Assistant entities, with no
manual YAML required:

1. Install the **Mosquitto broker** add-on (Settings → Add-ons → Add-on
   store) and set up the **MQTT** integration (discovery is enabled by
   default).
2. Restart Camon. `config.yaml` requests `mqtt:want` access to the
   Supervisor's broker service, so `run.sh` detects the Mosquitto add-on and
   configures `[mqtt]` for you automatically — you normally don't need to set
   anything under `[mqtt]` in `camon.toml` at all. (If you'd rather point at
   an external broker instead, set `[mqtt]` in `camon.toml` yourself; it's
   only overridden when a Supervisor-managed broker is actually installed.)

Each camera gets its own Home Assistant device, named **"Camon `<camera
id>`"**, containing:

- **A snapshot camera entity** — updated only while motion is being detected
  on that camera. This is by design: it keeps HA/the broker quiet for idle
  cameras instead of pushing an image on a fixed timer regardless of
  activity.
- **A motion `binary_sensor`** — on for the duration of a motion event.
- **One occupancy `binary_sensor` per detected object class** (from
  `analytics.object_detection.classes`, e.g. `person`, `car`, `truck`, `dog`,
  `cat`) — turns on when that class is seen and clears
  `occupancy_hold_secs` after the last sighting. Useful for automations like
  "only show the front-door camera card while a person is present."
- **One snapshot camera entity per detected object class** — shows the
  cropped frame from the last sighting of that class; retained, so it
  persists across restarts and keeps the picture long after the occupancy
  sensor has cleared.

Rename a camera in `camon.toml`, remove one, or drop a class from
`analytics.object_detection.classes`, and Camon removes the entities that went
with it on the next start — Home Assistant is told to forget them before the
device is marked available again, so no entity is left behind showing the
motion that was open when you edited the config. Classes are the careful case:
a start with object detection off, or with its vision server unreachable, keeps
announcing the occupancy entities — they simply read off — rather than deleting
them and their history. Removals are also scoped to the broker they were
announced to, so pointing Camon at a different broker never deletes entities
there on the strength of what the old one was told.

See `config.toml.example`'s `[mqtt]` section for the full set of keys (broker
host/port/credentials, topic prefixes, timing) if you need to override the
auto-configuration or point at an external broker.

## Building the image

The image is built by the `.github/workflows/addon.yml` workflow, which runs
whenever `camon-addon/` changes on `master` (or manually via
`workflow_dispatch`). It reads the version from `config.yaml`, builds the
`Dockerfile` with `camon-addon/` as the context, and pushes
`ghcr.io/nsg/amd64-addon-camon:<version>` (plus `:latest`) with the `io.hass.*`
labels Home Assistant expects. Releasing an add-on update is therefore: tag the
Camon release (`vX.Y.Z`), bump `version` in `config.yaml`, and push — the
workflow publishes the matching image tag, and installed add-ons see the
update.

The `Dockerfile` is multi-stage:

1. **Build** — `rust:1-bookworm`, `cargo build --release --locked`.
2. **Runtime** — `debian:bookworm-slim` with `ffmpeg` and `ca-certificates`
   installed; the compiled binary, `run.sh`, and the commented
   `config.toml.example` (seed template) are copied in.

It intentionally does **not** use a Home Assistant base image: those are
Alpine/musl, and a glibc binary from the Rust toolchain image won't run on
musl. Keeping both stages on Debian bookworm avoids a musl cross-compile.
`config.yaml` sets `init: false`, so the `CMD ["/run.sh"]` runs as PID 1 and
Camon receives `SIGTERM` directly for a graceful shutdown.

> The Dockerfile is exercised end-to-end by the GitHub Actions workflow on
> every change to `camon-addon/`; pull requests build the image without
> pushing, so a broken build fails CI before it can reach GHCR.
