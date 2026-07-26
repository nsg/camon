# Changelog

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
