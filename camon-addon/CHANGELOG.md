# Changelog

## 0.4.1

- **Fix MQTT auto-configuration: `curl` was missing from the 0.4.0
  image.** The Supervisor broker lookup failed with `curl: command not
  found` and fell back to "no Supervisor MQTT service found", so the Home
  Assistant entities never appeared unless `[mqtt]` was configured by hand
  in `camon.toml`. With this update the Mosquitto add-on is detected
  automatically as intended.

## 0.4.0

- **Native Home Assistant entities via MQTT discovery.** Install the
  Mosquitto broker add-on (plus the MQTT integration) and camon configures
  itself from the Supervisor — no settings needed. Each camera appears as
  its own Home Assistant device with:
  - a **snapshot camera** entity, updated only while motion is being
    detected (by design: idle cameras publish nothing),
  - a **motion** binary sensor, on for the duration of a motion event,
  - one **occupancy** binary sensor per object-detection class (e.g.
    `person`, `car`) that clears a configurable hold time after the last
    sighting — handy for showing a camera card only when a person is
    around, but not for every passing car.
  Without a broker the add-on runs exactly as before; an external broker
  can be configured under `[mqtt]` in `camon.toml`. See DOCS for details.
- **Camera ids are validated at startup when MQTT is enabled.** Ids
  containing the MQTT wildcards `+` or `#`, or two ids that normalize to
  the same slug (`Front Door` vs `front-door`), would silently break or
  collide in Home Assistant — camon now refuses to start with an error
  naming the camera to rename instead.
- **Motion analysis survives a frozen ffmpeg.** A decoder process that
  stalls (for example under severe host memory pressure) used to freeze
  motion detection silently while recording continued. Handing frames to
  the decoder is now bounded, and a decoder that consumes video without
  producing frames is caught by a tripwire — both cases kill and respawn
  the decoder automatically.
- Dependency refresh clearing all outstanding `cargo audit` advisories.

## 0.3.3

- **Event thumbnails are now color filmstrips for every event.** The crop
  decoder no longer stalls on ffmpeg's input-analysis buffering, frames are
  attached to the motion run that produced them, and extraction runs
  whenever events are recorded rather than only with object detection. The
  grayscale analysis-frame fallback is gone.

## 0.3.2

- **Memory fix:** GOP segments no longer carry Vec growth slack into the
  hot buffer, roughly halving steady-state RSS (about 560 MB per 6 Mbps
  camera instead of 1 GB). No functional changes.

## 0.3.1

- The add-on is now configured solely by a `camon.toml` file — the same
  format as a native install; nothing is set from the add-on's
  Configuration tab. camon gained the `--set` startup override the add-on
  uses to force its three container values, and JSON config support was
  removed.

## 0.3.0

- First Home Assistant add-on release: camon's web UI embedded in the
  sidebar via ingress.
- Stathost remote storage backend, streaming playback with HTTP Range
  support, TOML config with `--config`, detection mask, recovered-event
  badge, and a graceful-shutdown fix.
