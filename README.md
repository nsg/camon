<div align="center">
  <img src="logo.svg" alt="Camon" width="80">
  <h1>Camon</h1>
  <p>Multi-camera video surveillance with real-time analytics and tiered storage.</p>
</div>

---

## About

Camon is a self-hosted video surveillance system that ingests RTSP streams from IP cameras, runs real-time motion and object detection, and stores event footage to disk. It serves a web UI and REST API for live viewing, event browsing, and recorded playback — all from a single binary.

This is a personal project built for my own cameras, hardware, and use case. It is not designed to be general-purpose or easily adaptable to other setups. That said, if you find it useful and want to discuss making it work for your situation, feel free to open an issue — no promises, but happy to talk.

## Pipeline

We keep a single RTSP stream per camera, remux into MPEG-TS, segment at keyframe boundaries, and hold the last 10 minutes in a RAM buffer. We do that to avoid re-encoding to keep it lightweight on CPU usage, and we save it in RAM to spare disk wear.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/diagrams/01-capture-dark.svg">
  <img alt="Camera streams RTSP into the keyframe segmenter, which fills the 10-minute in-RAM hot buffer" src="docs/diagrams/01-capture-light.svg">
</picture>

Each camera runs its own independent pipeline with its own buffer. From the buffer we feed clients (browsers watching the live stream) and the motion analyzer. When a motion event ends, the analyzer pulls the complete event from the buffer and hands it to the warm writer, which persists it to disk right away. Segments aging out of the buffer are simply freed.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/diagrams/02-fanout-dark.svg">
  <img alt="The hot buffer serves HLS to clients and keyframes to the motion analyzer, which feeds the motion store, sends crop jobs to the detection worker (which fills the detection store and post-hoc-upgrades events), and hands finished events to the warm writer" src="docs/diagrams/02-fanout-light.svg">
</picture>

Clients can stream the raw segments from the buffer directly, the only thing we need to do is to serve a playlist.m3u8 which is a simple text file to the player. Object Detection is heavy on the CPU (or GPU) so most "detections" are filtered in the Motion Analyzer stage.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/diagrams/03-analyzer-dark.svg">
  <img alt="Keyframes are decoded to 320 by 240 grayscale and pass through the motion analyzer stages — MOG2 background subtraction, morphological opening, component filtering — into the motion store" src="docs/diagrams/03-analyzer-light.svg">
</picture>

To make decoding as lightweight as possible, we only extract keyframes because they are self-contained and do not depend on surrounding frames. With a 1-second GOP this means the analyzer effectively samples one frame per second. A segment carries its own answer for how many frames it owes — one per keyframe it holds — so a decode ends as soon as they arrive rather than when a timer runs out, and the timeout is left as the safety net it was meant to be. A segment whose frames never came produces no verdict at all rather than a quiet one: scoring unseen footage as motionless would both discard it and split a motion event across it. If the analyzer falls behind and segments age out of the hot buffer before it reaches them, the footage that went unanalyzed is reported rather than disappearing quietly.

The Motion Analyzer uses a built-in pure-Rust implementation of the Zivkovic MOG2 (Mixture of Gaussians) background subtractor — validated bit-exact against OpenCV's — to detect foreground motion, followed by morphological opening to eliminate noise and connected-component filtering to discard small blobs. The background model spans about 5 minutes at the 1 fps analysis rate, so persistent motion like tree sway gets absorbed into the background.

Motion detection is governed by three deterministic, per-camera controls, editable live from the web UI and persisted to `{data_dir}/{camera}/motion_settings.json`:

- **Sensitivity** — the MOG2 `var_threshold` (range 4–96, default 16; higher = less sensitive).
- **Min object size** — the minimum connected-component area in foreground pixels (range 50–2000, default 200), which discards small blobs like blowing leaves.
- **Movement mask** — a 16×12 grid of cells painted over the camera view; masked cells are zeroed in the MOG2 foreground mask before morphology and connected-component labeling, so they are excluded from detection deterministically. Read it as "nothing ever moves here": paint a busy road or a swaying tree to keep it from producing motion events.

There is no automatic tuning and no learned suppression: MOG2's verdict is the sole gate on what footage persists, so the settings only ever change when a human moves them. Config defaults for the two sliders can be set under `[analytics.motion]`; the per-camera file wins once a camera has been adjusted.

That file is written like an event: staged, fsynced, renamed into place, and the directory fsynced after — a truncated one loads as defaults, which would quietly un-paint a privacy mask — and saves are serialised per camera so two edits at once cannot share a staging file. A save that fails is reported rather than acknowledged: the API answers with the error and the web UI shows it above the settings panel. The change stays applied to the running detector either way, because a mask exists to stop something being seen and has to take effect even when the disk will not take it; what is lost is only that it survives a restart.

A fourth per-camera control, the **detection mask**, is painted on the same 16×12 grid but works one stage later and independently of motion detection. Its cells read as "the vision model never sees these pixels": every cell painted here is blacked out of every frame handed to the Ollama vision model before JPEG encoding, in the crop's own coordinate space (the cell rectangle is intersected with each crop and translated, so masked pixels are removed no matter how the frame was cropped — including the full-frame crop a lighting change can force). Motion detection is untouched; only classification is suppressed. Use it for a stationary nuisance object — a parked car that would otherwise be reported as "car" whenever a full-frame crop briefly includes it. The web UI's mask editor paints both layers on one grid, switching between the movement mask (red) and detection mask (orange) with a layer toggle. The detection mask defaults to all-off and is persisted alongside the other settings in `motion_settings.json`.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/diagrams/04-detection-dark.svg">
  <img alt="A motion event is subsampled to four frames, cropped using motion bounding boxes, JPEG-encoded, and queued to the single global serial detection worker, which talks to Ollama one request at a time, records verdicts in the detection store, and upgrades events via the warm writer" src="docs/diagrams/04-detection-light.svg">
</picture>

The Motion Store keeps track of motion events. If there are several segments in sequence that have movements, they are considered a single motion event. We sample four frames from each event at 0/3, 1/3, 2/3 and 3/3. Using bounding boxes from the motion event, we crop the image to "zoom in" to the action, JPEG-encode the crops, and enqueue them as a job for a single global detection worker shared by all cameras. The worker is strictly serial — at most one in-flight Ollama request at any time — so a modest GPU is never hit with parallel load, and the analyzer never waits for the model: if the small queue is full the job is simply dropped with a warning (the motion event still records; only the object classification is lost).

The model is asked for structured output: the request carries a JSON schema (via Ollama's `format` field) with the configured class list as an enum, so the response is machine-parseable JSON with per-detection class, confidence, and a normalized bounding box. Responses are validated — out-of-range confidences, garbage boxes, and unknown classes are dropped. Every valid detection is recorded — there is no learned suppression that could silently drop a real detection. To stop a persistently busy area (a tree, a road) from generating events, paint it into the per-camera movement mask; to stop a stationary object (a parked car) from being classified without suppressing motion there, paint it into the per-camera detection mask, which blacks those cells out of every frame sent to the vision model.

We are still running in RAM, our 10 minute video buffer and some metadata stored in the motion and detection stores. At this stage we should have enough information to only save events that we care about on disk!

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/diagrams/05-storage-dark.svg">
  <img alt="At event end the motion analyzer assembles segments plus motion and detection metadata and hands the event to the warm writer, which persists movements, objects, metadata, and thumbnails into warm storage (local disk or stathost); clients stream from storage via HLS and pruning removes old events" src="docs/diagrams/05-storage-light.svg">
</picture>

The moment a motion run ends (post-padding elapsed), the analyzer assembles the complete event — pre-padding, motion, and post-padding pulled straight from the hot buffer, plus metadata from the motion and detection stores — and hands it to the warm writer, which persists it to disk immediately. This way an event only stays at risk in RAM for seconds after it ends, not until its segments age out of the buffer. The writer also saves metadata and thumbnails that is used by the web UI. User can stream saved video events via HLS.

Event writes never wait for the vision model. If an Ollama verdict arrives while the run is still open, it is picked up during assembly as before; if it arrives after the event is already on disk, the detection worker asks the warm writer to upgrade it post-hoc — the sidecar is rewritten with the detections and the files move from `movements/` to `objects/`, which switches the event to the longer object retention. All file mutations go through the warm writer, so writes and upgrades can never race. If a verdict lands in the tiny window while the event is being assembled, worst case the event simply stays movement-classified with the detections still visible in the detection store and API.

### Recording modes

The `storage` and `analytics` flags together select what gets recorded:

- **Event recording** (`storage.enabled = true`, `analytics.enabled = true`) — the default. Only motion and object events are saved, into `movements/` and `objects/`. Storage is proportional to activity.
- **Continuous recording** (`storage.enabled = true`, `analytics.enabled = false`) — "dumb NVR" mode. With no analyzer to gate on motion, a per-camera recorder rolls every segment from the hot buffer to disk in fixed-length chunks (each `max_event_duration_secs` long) under `continuous/`. Every chunk is a whole-GOP `.ts` that decodes on its own, and successive chunks carry a `"continues"` flag so the timeline stitches seamlessly. This is heavy — roughly **43 GB/day/camera at 4 Mbps** — so `continuous_retention_days` defaults to 1.
- **No recording** (`storage.enabled = false`) — live view only; nothing hits disk. camon says which of the two storage-off states it is in at startup: with analytics still on it **warns**, because motion is detected and published to MQTT and then thrown away, which is a deliberate setup for a motion sensor and an expensive surprise if recording was the point.

Both modes share the same warm writer, retention pruning, and HLS playback path; only the trigger differs.

### Durability

Event files are written atomically — staged as `.tmp`, fsynced, renamed into place, and the containing directory fsynced after — so a crash or power cut never leaves a half-indexed event, and an event camon has reported as written stays written. That last fsync is what makes the rename count: renaming is atomic but not durable on its own, so without it a power cut can lose the directory entry naming the file while the bytes sit intact and unreachable on the disk. The sidecar and thumbnails, written before the commit, are covered by the same directory fsync; their *contents* are deliberately left unsynced, because a lost detection can be recovered from the detection store and the footage cannot. A movement→object upgrade is committed the same way, on both directories it moves the event between — destination first, so a failure there stops the sequence rather than making the old name's removal durable while the new one is not. When a directory fsync does fail, the event is still on disk, indexed and playable, but the write is **reported as failed**: nothing is re-written and nothing is dropped by saying so, and a camera whose commits are all failing to sync must not go on looking healthy to the recording-silence watchdog. On the next startup camon **recovers** interrupted writes instead of discarding them: an orphaned `.ts.tmp` (which may hold the footage of exactly the incident that cut the power) is trimmed to its last intact packet, its real duration recomputed from the stream timestamps, and indexed like any other event with a `"recovered": true` flag. If the disk runs low, a `min_free_bytes` guard emergency-prunes the oldest recordings (continuous first, then movements, then objects) so the writer keeps recording instead of failing.

Retention belongs to one task for the whole store rather than to each camera's writer — a sweep covers every camera, so one task does it however many cameras are configured, instead of a sweep per camera racing all the others — and it runs hourly; on shutdown it stops between events, never part-way through deleting one. An event leaves the index only once its video is actually gone — a delete that fails leaves the event listed and playable and the next sweep retries it, instead of unindexing a recording that is still occupying disk with nothing left to describe it. The emergency low-space pass skips events it has already failed to delete, so they cannot stand in front of events that would free space, and space it failed to reclaim is never counted as reclaimed.

A scheduled sweep also deletes at most a quarter of a camera's events (at least four, so a small archive still drains). An event's age is measured against the wall clock, and a box without a battery-backed clock resumes at its last shutdown time and is then corrected forward by however long it was switched off — which makes the whole archive look expired at once. The cap turns that into a loud warning and a trickle rather than an empty archive; the held-back events follow on later sweeps, and ordinary expiry is a few percent of an archive per sweep and never reaches the limit. It bounds the scheduled sweep only: a disk genuinely running out still prunes as much as it needs, as does the stathost `max_stored_bytes` budget.

A shutdown — from SIGTERM or from an installed update — flushes what is in flight, and nothing in it waits out a retry: a camera parked in its reconnect backoff and an analyzer waiting to respawn its decoder both wake the moment the shutdown is requested, so stopping camon never costs a minute of waiting for work that has already been abandoned.

A camera can also end up recording nothing without anything crashing — an ignore mask painted over the whole frame, a sensitivity slider at its least sensitive, a stream that never reaches camon, or writes that keep failing. Each of those looks identical from the outside: an empty event list. camon therefore **watches each camera for silence** and warns when one has written no event for too long. The limit depends on what the camera is supposed to produce: a continuous recorder rolls a chunk every `max_event_duration_secs` whatever the scene does, so ten missed chunks in a row (20 minutes at the default) is already a fault, while in event mode an empty garden legitimately scores no motion all night and the limit is 24 hours — half the default `movement_retention_days`, so the warning arrives while there is still footage to save. The silence is measured from the newest event **on disk**, not from process start, so it survives restarts, and each warning states the whole silence rather than the time since the last one. Cameras that are not expected to record (storage disabled) are not watched at all.

Knowing a camera is silent is not the same as knowing why, so each connection also reports what camon actually saw on it: whether bytes arrived at all, whether they were a transport stream, whether a program map named an H.264 stream, whether packets reached that stream, and whether any of them flagged a random access point. That last one is the quiet killer — camon cuts a segment only on that flag, and a camera that never sets it produces a stream that looks perfectly alive and yields no footage, which without the report would show up as nothing but an endless reconnect loop. Each report says what was observed and over how long and offers the causes that fit rather than naming one, a connection too short to judge says so instead of blaming the camera, and repeated failures escalate on a widening schedule instead of repeating every minute. The count clears only when a connection actually records something.

### Storage backends

The warm event store sits behind a backend seam, so *where* events live is a config choice — analytics, motion settings, and the in-RAM hot buffer always stay local regardless.

- **Local disk** (default) — the `data_dir` on the machine running camon, with the atomic write ladder, crash recovery, and `min_free_bytes` free-space guard described above.
- **Remote stathost** — a [stathost](https://github.com/nsg/stathost) static file host, selected by adding a `[storage.stathost]` section. Each event becomes three sibling objects under `{camera}/` on the host — the `.ts` video, a `.json` sidecar (which here also carries the event **type**, since there are no `movements/`/`objects/`/`continuous/` directories), and eager `{stem}_thumb_{i}.jpg` filmstrip frames. Time-based retention still applies via `DELETE`; because the client can't see the server's disk, retention-by-space becomes a client-side **budget** (`max_stored_bytes`) that prunes the oldest events (continuous → movements → objects) when tracked usage exceeds it. Post-hoc movement→object upgrades just rewrite the sidecar in place — no object is ever renamed or moved.

  Every request to the host is bounded, because the warm writer awaits them inline and an unbounded one would wedge that camera's recording and its shutdown: 10s to connect, 60s in total for a delete, a sidecar or thumbnail read, or the startup listing, and 300s for an upload, which has to allow tens of megabytes on a slow uplink. Ranged playback is the deliberate exception — a total ceiling would cut off a player that drains the body at its own pace, so it is bounded by a 30s idle budget instead.

  Requires stathost **0.2.0 or later** — camon relies on its atomic uploads (readers never see a partially uploaded object), its detailed listing (`?detail=true`) for accurate storage accounting, and its HTTP Range support; there is no fallback for older servers. Because the sidecar is the only record of an event's **type**, it is uploaded *before* the video, which commits the event — an event whose sidecar cannot be stored is failed rather than written as a video that would scan back as a plain movement and expire on the wrong retention. The one exception is a plain movement event, which is exactly what a sidecar-less `.ts` scans back as anyway. Nothing is rolled back on a failed upload (a failure can still have committed server-side): the leftovers are an orphan sidecar, which the scan ignores, or a video that its sidecar still types correctly. Warm-event playback **streams** rather than buffering the whole event in RAM: the playback handler forwards a single `Range` header to stathost and relays the `206`/`Content-Range` response, so seeking fetches only the requested bytes.

  ```toml
  [storage.stathost]
  url = "https://files.example.com"
  bucket = "camon"
  token = "s3cr3t-token"
  max_stored_bytes = 0   # 0 = unlimited; rely on time-based retention
  # enabled = true       # set false to keep this section but use local disk
  ```

## Quick Start

Camon is a single self-contained binary — the only runtime dependency is FFmpeg. Install it and download the latest binary from [GitHub Releases](https://github.com/nsg/camon/releases):

```bash
sudo apt install ffmpeg

curl -fLO https://github.com/nsg/camon/releases/latest/download/camon-linux-glibc
chmod +x camon-linux-glibc
./camon-linux-glibc
```

Camon loads `config.toml` from the current working directory. `camon version` — or `--version` / `-V`, recognised as the **first** argument, since anywhere else they are unknown flags and ignored — prints the release version and the exact build it came from, and does nothing else: it answers before the config is even read, because the self-updater uses it to ask a freshly downloaded binary what it really is.

> **Note:** Pre-built binaries are linked against glibc. musl-based systems are not supported.

### Building from Source

Building needs nothing beyond a stable Rust toolchain (no OpenCV, no C++ toolchain — the vision code is pure Rust); ffmpeg is only needed at runtime:

```bash
sudo apt install ffmpeg
cargo build --release
./target/release/camon
```

## Configuration

Create a `config.toml` in the working directory. Point Camon at a config anywhere with `--config <path>`, and override individual values at startup with one or more `--set <dotted.path>=<value>` flags (overrides win over the file). The value is typed from the setting it names rather than from how it looks, so a numeric-looking string stays a string — `--set http.port=8080` sets a number, `--set mqtt.password=8080` sets text:

```bash
camon --config /etc/camon/config.toml --set http.port=9090 --set update.enabled=true
```

The whole configuration is validated before anything starts, and camon exits with the problem rather than running on defaults it silently fell back to — an unknown key included, since a typo is otherwise indistinguishable from an omission (keys camon itself shipped and later removed are dropped with advice instead, so a config that booted before keeps booting).

All sections are optional — defaults are shown below:

```toml
[update]
# Auto-update from GitHub Releases on startup and every 12 hours
# (default: false — opt in by setting this to true).
# A downloaded binary must be a valid ELF and must match the release's
# sha256sums.txt; a release that publishes no sha256sums.txt is refused rather
# than installed unverified. That is corruption protection, not a security
# guarantee — whoever can swap the asset can swap the checksums — and since the
# installed service runs as root and nothing is signed, updating is off unless
# you ask for it.
# Because the restart is what completes an update, a bad release must not be
# able to turn it into a loop, so two things bound it. First, the download is
# asked what version it is — run as `camon version` while still staged beside
# the binary it would replace, with its output, runtime and process group all
# bounded — and it must report exactly the version its release is tagged with.
# A release whose tag does not match its asset is refused, since installing it
# would leave the tag looking newer and camon would fetch it again after every
# restart. A binary that cannot answer is refused too. Refusals are recorded,
# so the asset is not re-downloaded every 12 hours to reach the same verdict.
# Second, camon counts installs in a `camon.update-guard` file beside the
# binary and stops after three of the same version. That covers the case the
# version check cannot see: an install that works but never comes back, because
# the service starts a different binary than the updater replaces. Both
# refusals are logged loudly with what to fix; deleting the guard file retries,
# and any newer release resets it, so nothing can hold a genuine update back.
# One updater at a time touches an installation (a lock file beside the
# binary), and the release tag must equal the version in the binary's
# Cargo.toml — the release workflow checks this.
# An update installed while camon is running triggers the same
# graceful shutdown as SIGTERM — recordings in flight are flushed — and the
# service manager starts the new binary: `camon install service` writes a
# systemd unit with Restart=always, or an OpenRC script supervised by
# supervise-daemon. Started any other way, camon just exits after an update and
# stays down. A drain still unfinished six minutes after the update is
# abandoned: that guarantees the restart eventually happens even if the drain
# has wedged, at the cost of losing what was left to write — which against a
# very slow remote warm-storage server can be a drain that was still making
# progress.
enabled = false

[buffer]
# Hot buffer duration in seconds (default: 600 = 10 minutes)
hot_duration_secs = 600

[http]
# HTTP server port for web UI and API (default: 8080)
port = 8080
# Address to bind to (default: "0.0.0.0" = every interface, reachable from the
# LAN). "127.0.0.1" keeps the server on this machine only.
bind = "0.0.0.0"
# Access token for the API (default: none = anyone who can reach the port can
# watch all footage and change motion settings). When set, every request under
# /api must present it as an "Authorization: Bearer <token>" header; GET and
# HEAD may instead use a "?token=<token>" query parameter, which is how image
# and video URLs that cannot set headers authenticate. The web UI prompts for
# it once and keeps it in the browser's local storage; the UI shell itself
# stays unauthenticated so that prompt can load.
# token = "s3cr3t-token"
# Silence the startup warning about an unauthenticated API on a non-loopback
# bind (default: false). Set true only when something in front of camon already
# authenticates — e.g. the Home Assistant add-on, where ingress does.
allow_open = false

[analytics]
# Enable MOG2 motion detection (default: false)
enabled = true
# Frame rate ffmpeg decodes at when extracting frames from a motion run for
# object detection (default: 5). Motion analysis itself is keyframe-driven —
# roughly one frame per second at a 1-second GOP — and ignores this.
sample_fps = 5

# Object detection via Ollama (requires analytics enabled).
# One global worker serves all cameras, strictly serially (max one in-flight
# request); events never wait for the model.
[analytics.object_detection]
# Enable object detection on motion segments (default: false)
enabled = true
# Minimum confidence threshold (default: 0.5)
confidence_threshold = 0.5
# Object classes to detect. Omit the key for the defaults (person, car,
# truck, dog, cat). While detection is enabled the list may not be empty and
# no entry may be blank — "detect nothing" is what enabled = false says.
# Also constrains the model's structured JSON output. Trimmed, lower-cased
# and deduplicated at load, and while [mqtt] is enabled a class may not
# contain "+" or "#" — it reaches the occupancy topic verbatim.
classes = ["person", "car", "truck", "dog", "cat"]

[analytics.object_detection.ollama]
# Ollama server URL (default: http://localhost:11434)
url = "http://localhost:11434"
# Vision model to use (default: gemma4:e4b)
model = "gemma4:e4b"
# Per-request timeout in seconds (default: 90). A timeout costs only that
# event's object upgrade, never the footage.
timeout_secs = 90

# Optional fallback server if primary fails
# [analytics.object_detection.ollama.fallback]
# url = "http://backup:11434"
# model = "gemma3:4b"

[storage]
# Enable warm storage — write video to disk (default: true).
# With analytics enabled this is EVENT recording (motion/object events only).
# With analytics DISABLED it becomes CONTINUOUS recording ("dumb NVR"): every
# segment is saved — roughly 43 GB/day/camera at 4 Mbps.
enabled = true
# Directory for event files (default: /var/camon/storage)
data_dir = "/var/camon/storage"
# Seconds of context before first motion in an event (default: 5)
pre_padding_secs = 5
# Seconds of context after last motion in an event (default: 10)
post_padding_secs = 10
# Cap on a single event's length in seconds (default: 120). Longer runs are
# split into chained, independently playable chunks (follow-ons flagged
# "continues"). In continuous mode this is the length of each chunk. A recording
# must fit inside buffer.hot_duration_secs: cap + pre_padding_secs with analytics
# on (a warning), the cap alone in continuous mode (an error). 0 disables
# chunking — fine with analytics on, rejected in continuous mode.
# Timing is monotonic, immune to camera PTS jumps.
max_event_duration_secs = 120
# Retention for movement-only events in days (default: 2). All three retentions
# are whole days from 1 to 3650; 0 is rejected, as it would expire every event
# the moment it is written. One task sweeps every camera hourly, and a sweep
# deletes at most a quarter of a camera's events, so a clock corrected forward
# cannot empty the archive in one pass — see "Durability" above.
movement_retention_days = 2
# Retention for object detection events in days (default: 14)
object_retention_days = 14
# Retention for continuous-recording chunks in days (default: 1). Short because
# continuous recording is ~43 GB/day/camera at 4 Mbps.
continuous_retention_days = 1
# Minimum free space (bytes) on the storage filesystem (default: 2 GiB).
# Below this, the oldest recordings are emergency-pruned (continuous →
# movements → objects) before each write. 0 disables the guard. This pass skips
# events it has already failed to delete and is not bound by the sweep's cap.
min_free_bytes = 2147483648

# Optional: send warm events to a remote stathost host instead of data_dir.
# See "Storage backends" above. When present and enabled, min_free_bytes is
# ignored in favour of the max_stored_bytes budget.
# [storage.stathost]
# url = "https://files.example.com"
# bucket = "camon"
# token = "s3cr3t-token"
# max_stored_bytes = 0

[mqtt]
# MQTT bridge to Home Assistant (default: false). When enabled, camon
# publishes MQTT discovery messages creating one HA device per camera: a
# snapshot camera entity (updated only while motion is detected — idle
# cameras stay quiet), a motion binary_sensor, and per
# analytics.object_detection.classes entry both an occupancy binary_sensor
# that clears occupancy_hold_secs after the last sighting and a snapshot
# camera showing the cropped frame from the last sighting of that class
# (retained, so it persists across restarts). In the Home Assistant add-on
# this section is auto-configured from the Mosquitto add-on via the
# Supervisor and normally needs no manual settings. On every (re)connect camon
# restates every configured entity explicitly, on or off, and flips
# availability to "online" last, so a retained ON left behind by a camon that
# died mid-motion is always contradicted. Snapshots are not queued while the
# broker is unreachable — an image would be long superseded by the time it
# could be delivered — and each snapshot decode gives up after 15 seconds.
# While enabled, camera ids must not contain "+" or "#" (MQTT wildcards) and
# must stay unique once lowercased with non-alphanumerics folded to "_"; the
# same wildcard rule applies to the object detection class names, since they
# reach the topic verbatim. camon refuses to start otherwise.
enabled = false
# Broker hostname and port (defaults: "localhost", 1883)
host = "localhost"
port = 1883
# Broker credentials (optional, default: none)
# username = "camon"
# password = "s3cr3t"
# Prefix for camon's own state topics (default: "camon")
topic_prefix = "camon"
# Prefix the HA MQTT integration watches for discovery (default: "homeassistant")
discovery_prefix = "homeassistant"
# Snapshot cadence while motion is active, in seconds (default: 5)
snapshot_interval_secs = 5
# Seconds an occupancy sensor stays "on" after the last sighting (default: 60)
occupancy_hold_secs = 60

# Add one [[cameras]] block per camera. Each id must be unique and is used
# verbatim as a directory name under storage.data_dir, so it must be a single
# path component: no "/", "\" or NUL, not "." or "..", not blank, no
# leading/trailing whitespace. Accents and punctuation are fine ("Trädgård").
[[cameras]]
id = "front-door"
url = "rtsp://admin:password@192.168.1.100:554/stream1"
```

### Camera Requirements

- RTSP H.264 stream at 1080p 30fps
- GOP (keyframe interval) of 1–2 seconds
- Bitrate ~6 Mbps (CBR or capped VBR)

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/cameras` | List configured cameras |
| `GET` | `/api/stream/{id}/playlist.m3u8` | Live HLS playlist |
| `GET` | `/api/stream/{id}/segment/{n}` | Live HLS segment |
| `GET` | `/api/cameras/{id}/motion` | Motion segments with timestamps |
| `GET` | `/api/cameras/{id}/motion/{seq}/mask` | JPEG motion mask for a segment |
| `GET` | `/api/cameras/{id}/motion/stability` | JPEG final motion mask (after component filtering) |
| `GET` | `/api/cameras/{id}/motion/stability/raw` | JPEG raw MOG2 foreground mask |
| `GET` | `/api/cameras/{id}/motion/stability/no-shadow` | JPEG alias of the raw mask (shadow stage removed) |
| `GET` | `/api/cameras/{id}/motion/stability/morph` | JPEG after morphological opening |
| `GET` | `/api/cameras/{id}/motion/background` | JPEG learned background model |
| `GET` | `/api/cameras/{id}/motion/settings` | Motion settings: sensitivity, min object size, movement mask, detection mask (JSON) |
| `PUT` | `/api/cameras/{id}/motion/settings` | Update motion settings (partial JSON: `var_threshold`, `min_contour_area`, `mask`, `detection_mask`) |
| `GET` | `/api/cameras/{id}/detections` | Detected objects with confidence |
| `GET` | `/api/cameras/{id}/detections/{id}/frame` | JPEG frame of detection |
| `GET` | `/api/cameras/{id}/hot-events` | Hot buffer motion events |
| `GET` | `/api/cameras/{id}/events` | Warm events overlapping a time range (`from`, `to`) |
| `GET` | `/api/cameras/{id}/events/{pts}/playlist.m3u8` | Warm event HLS playlist |
| `GET` | `/api/cameras/{id}/events/{pts}/segment` | Warm event HLS segment |
| `GET` | `/api/cameras/{id}/events/{pts}/thumbnail` | Warm event thumbnail JPEG |
| `GET` | `/api/cameras/{id}/events/{pts}/filmstrip/{index}` | Filmstrip frame JPEG |
| `GET` | `/api/cameras/{id}/detection-debug` | Detection debug entries |
| `GET` | `/api/cameras/{id}/detection-debug/{id}/frame/{index}` | Detection debug frame JPEG |
| `GET` | `/api/cameras/{id}/detection-debug/{id}/full-frame` | Detection debug full frame JPEG |

Every route above sits behind `[http] token` when one is set: a request without it answers `401` with `WWW-Authenticate: Bearer`. Only `/api` is covered — the UI shell (`/`) and its assets stay open so the token prompt can load. A GET may carry the token as `?token=` instead of the header; a `PUT` may not.

`from` and `to` on the event query are wall-clock nanoseconds, and either may be left out (an omitted `from` reaches back to the start of the archive, an omitted `to` runs to the end). The range is matched by **overlap**, so an event that started before `from` and is still running inside it is returned; that is how a long continuous chunk shows up in a window it merely spans. A range with `from` greater than `to` answers `400` rather than guessing at what was meant.

`PUT /api/cameras/{id}/motion/settings` answers `500` when the new settings could not be written to disk. The change is still applied to the running detector — a mask exists to stop something being seen, so it has to take effect even when the disk will not take it — and the body says so; what is lost is only that it survives a restart.

## Storage Tiers

| Tier | Medium | Retention | Quality | Purpose |
|---|---|---|---|---|
| Hot | RAM | ~10 minutes | 1080p @ 30fps | Live playback and real-time analysis |
| Warm | Disk | 2 days (movement) / 14 days (objects) / 1 day (continuous) | Original quality | Motion-triggered events, or gapless continuous recording when analytics is off |

## Home Assistant add-on

This repository is also a valid [Home Assistant add-on repository](https://developers.home-assistant.io/docs/add-ons/repository), so Camon can be installed as an add-on with its web UI embedded in the Home Assistant sidebar (via ingress).

[![Add repository to my Home Assistant](https://my.home-assistant.io/badges/supervisor_add_addon_repository.svg)](https://my.home-assistant.io/redirect/supervisor_add_addon_repository/?repository_url=https%3A%2F%2Fgithub.com%2Fnsg%2Fcamon)

See the [add-on documentation](camon-addon/DOCS.md) for install steps, ingress notes, the `camon.toml` configuration, and the amd64-only caveat.

With the `[mqtt]` section enabled (automatic in the add-on when the Mosquitto broker add-on is installed), each camera also appears as native Home Assistant entities via MQTT discovery: a motion-gated snapshot camera, a motion sensor, per-class occupancy sensors, and a per-class snapshot camera showing the cropped frame from the last sighting of that class (retained, so it persists across restarts) — no custom integration required.

Because every state is published retained, a connection that drops mid-motion would otherwise leave Home Assistant showing movement that never ends. Every reconnect therefore restates *every* configured entity — the ones that are off just as loudly as the ones that are on — and only then marks the device available again, so nothing is trusted while the broker still holds a stale value.

## License

MIT — see [LICENSE.md](LICENSE.md).
