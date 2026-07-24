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
4. Create the configuration file (see [Configuration](#configuration) below),
   then **Start**. With ingress enabled, **Camon** shows up in the sidebar —
   click it to open the web UI.

## Repository layout

Standard HA add-on repository conventions:

```
repository.yaml        # repository manifest (name/url/maintainer) at repo root
camon-addon/           # one add-on == one top-level folder
  config.yaml          # add-on manifest (slug, arch, ingress, map)
  Dockerfile           # multi-stage: build Camon from source, run on Debian+ffmpeg
  run.sh               # points Camon at /config/camon.toml (forcing 3 values), then exec camon
```

The Home Assistant supervisor builds the image with `camon-addon/` as the
Docker context, so the Rust source is not in the context — the build stage
instead clones this repository at the version tag matching `config.yaml`
(`v0.3.1`), so the add-on always compiles the exact released source. (That tag
must exist when the image is built — tag the release before installing.)

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
available — see [`config.toml.example`](config.toml.example) for the full,
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

Three values are **forced by `run.sh`** via `camon --set`, overriding whatever
`camon.toml` says — they are non-negotiable inside the container:

- **`update.enabled = false`.** Camon's built-in GitHub self-updater is
  disabled inside the add-on. The container filesystem is ephemeral and updates
  must flow through the Home Assistant add-on store, so an in-container binary
  swap would be pointless (lost on rebuild) and would fight Home Assistant's
  own update mechanism.
- **`http.port = 22666`.** The add-on is reached exclusively through ingress,
  which is wired to internal port 22666 — an uncommon port picked so it can
  never collide with another service even if host networking is ever enabled
  (with ingress-only access, no host port is bound at all). The port is pinned
  so that wiring can't be broken from the config file.
- **`storage.data_dir = /data/storage`.** `/data` is the add-on's own
  persistent volume — always mounted and preserved across restarts/updates — so
  recordings survive there automatically.

Everything else — analytics, object detection, the Ollama server (including the
fallback), retention, motion tuning, remote stathost storage, and the camera
list — is set in `camon.toml` exactly as documented for a native install.

## Building the image

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

> The Dockerfile and `config.yaml` here were validated by conformance to the
> Home Assistant add-on docs; they were **not** built/run in CI in this
> environment (no Docker available), so treat the first `ha` build as the real
> smoke test.
