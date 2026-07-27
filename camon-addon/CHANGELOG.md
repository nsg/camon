# Changelog

## Unreleased

- **`camon.toml` is now checked strictly at startup, and a mistake stops
  the add-on instead of being ignored.** Previously any key camon did not
  recognise was skipped in silence, so a typo like
  `movment_retention_days` left the default quietly in force and looked
  like a setting that did nothing. Every unknown key is now an error
  naming the key, on the first start after this update. If the add-on
  stops with a config error, fix the named key in
  `/addon_configs/<repo>_camon/camon.toml` and start it again.
  - Keys from camon 0.2.0 and earlier are the exception: `backend` and
    `model_path` under `[analytics.object_detection]` were removed when
    object detection became Ollama-only, and are ignored with a warning in
    the log rather than treated as errors. Delete them at your
    convenience; the model is set with `model` under
    `[analytics.object_detection.ollama]`.
- `storage.max_event_duration_secs = 0` is no longer accepted when
  `analytics.enabled = false`. It half-worked before: with analytics on it
  means "never split an event", which is still allowed, but in continuous
  recording it left nothing to roll a chunk, so no footage was written
  until the add-on shut down.
- A recording that cannot fit in the hot buffer is now called out at
  startup. In continuous recording, `storage.max_event_duration_secs` must
  be below `buffer.hot_duration_secs` or the add-on stops — nothing rolls
  a chunk, so no footage would be written at all. With analytics enabled
  it is a warning in the log instead of an error: recording continues,
  but events running longer than roughly `hot_duration_secs` minus
  `pre_padding_secs` lose their opening seconds. Either way the message
  names the values involved.
- Camera ids are validated: they must be unique and usable as a folder
  name (no `/` or `\`, not `.` or `..`, not blank or space-padded).
  Accented and punctuated names such as `Trädgård` or `Garage (side)`
  work as before. Duplicate ids used to start two recorders writing over
  each other in one folder.
- Object classes in `analytics.object_detection.classes` are lowercased
  and deduplicated when loaded, so `classes = ["Person"]` now creates a
  working occupancy sensor instead of one that could never turn on.

## 0.5.0

- **The live view's detection surfaces merge into one timeline.** Motion
  history draws directly on the scrubber as an intensity histogram, and
  object detections sit on top as markers — tap one for the classified
  crop with confidence, tap again to jump the video there. Dragging
  anywhere on the track scrubs the hot buffer. Built phone-first, with
  proper touch targets; the separate detection gallery and recent-motion
  chips are gone.
- **Stored events get per-day activity maps** below the live view: one
  24-hour strip per day showing when movements (amber) and objects (red)
  happened. Tapping a day opens the event browser at that day. Days follow
  whatever the server has retained, so different movement/object retention
  settings just work.
- Live playback that quietly fell behind after a player stall now snaps
  back to the live edge instead of drifting minutes into the past.

## 0.4.2

- **Every occupancy sensor now has a matching snapshot camera** (e.g. "Cat
  snapshot" next to "Cat occupancy"): the cropped frame the vision model
  actually classified when it last saw that class. Retained on the broker,
  so the tile keeps showing the last sighting — long after the occupancy
  sensor has cleared, and across Home Assistant restarts. Handy for
  notifications that show *the cat*, not just the camera.

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
