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

```mermaid
flowchart LR
  Camera -->|RTSP| KeyframeSegmenter("Keyframe Segmenter") --> HotBuffer[("Hot Buffer (10m)")]
```

Each camera runs its own independent pipeline with its own buffer. From the buffer we feed clients (browsers watching the live stream) and the motion analyzer. When a motion event ends, the analyzer pulls the complete event from the buffer and hands it to the warm writer, which persists it to disk right away. Segments aging out of the buffer are simply freed.

```mermaid
flowchart LR
  HotBuffer[("Hot Buffer (10m)")] -->|HLS| Clients
  HotBuffer -->|keyframes| MA["Motion Analyzer"] --> MS[("Motion Store")]
  MA -->|crop jobs| OD["Detection Worker (Ollama)"] --> DS[("Detection Store")]
  MA -->|finished events| WW[("Warm Writer (disk)")]
  OD -.->|post-hoc upgrades| WW
```

Clients can stream the raw segments from the buffer directly, the only thing we need to do is to serve a playlist.m3u8 which is a simple text file to the player. Object Detection is heavy on the CPU (or GPU) so most "detections" are filtered in the Motion Analyzer stage.

```mermaid
flowchart LR
  HotBuffer[("Hot Buffer (10m)")] -->|keyframes| Decode["Decode 320x240 (grayscale)"] --> MA
  MA -.-> AutomaticTuning("Automatic Tuning") -.-> MA
  MA --> MS[("Motion Store")]

subgraph MA["Motion Analyzer"]
   MOG2["Background Subtraction"] --> Morph["Morphological Opening"] --> Components["Component Filtering"]
end
```

To make decoding as lightweight as possible, we only extract keyframes because they are self-contained and do not depend on surrounding frames. With a 1-second GOP this means the analyzer effectively samples one frame per second.

The Motion Analyzer uses a built-in pure-Rust implementation of the Zivkovic MOG2 (Mixture of Gaussians) background subtractor — validated bit-exact against OpenCV's — to detect foreground motion, followed by morphological opening to eliminate noise and connected-component filtering to discard small blobs. The background model spans about 5 minutes at the 1 fps analysis rate, so persistent motion like tree sway gets absorbed into the background.

Motion detection is governed by three deterministic, per-camera controls, editable live from the web UI and persisted to `{data_dir}/{camera}/motion_settings.json`:

- **Sensitivity** — the MOG2 `var_threshold` (range 4–96, default 16; higher = less sensitive).
- **Min object size** — the minimum connected-component area in foreground pixels (range 50–2000, default 200), which discards small blobs like blowing leaves.
- **Movement mask** — a 16×12 grid of cells painted over the camera view; masked cells are zeroed in the MOG2 foreground mask before morphology and connected-component labeling, so they are excluded from detection deterministically. Read it as "nothing ever moves here": paint a busy road or a swaying tree to keep it from producing motion events.

There is no automatic tuning and no learned suppression: MOG2's verdict is the sole gate on what footage persists, so the settings only ever change when a human moves them. Config defaults for the two sliders can be set under `[analytics.motion]`; the per-camera file wins once a camera has been adjusted.

A fourth per-camera control, the **detection mask**, is painted on the same 16×12 grid but works one stage later and independently of motion detection. Its cells read as "the vision model never sees these pixels": every cell painted here is blacked out of every frame handed to the Ollama vision model before JPEG encoding, in the crop's own coordinate space (the cell rectangle is intersected with each crop and translated, so masked pixels are removed no matter how the frame was cropped — including the full-frame crop a lighting change can force). Motion detection is untouched; only classification is suppressed. Use it for a stationary nuisance object — a parked car that would otherwise be reported as "car" whenever a full-frame crop briefly includes it. The web UI's mask editor paints both layers on one grid, switching between the movement mask (red) and detection mask (orange) with a layer toggle. The detection mask defaults to all-off and is persisted alongside the other settings in `motion_settings.json`.

```mermaid
flowchart LR
  HotBuffer[("Hot Buffer (10m)")] -->|motion event| Sub["Subsample 4 Frames"]
  MA[("Motion Store")] -->|bounding boxes| Sub
  Sub --> Crop["Crop + JPEG"] -->|bounded queue| Worker["Detection Worker (global, serial)"]
  Worker -->|one request at a time| Ollama["Ollama"]
  Worker --> DS[("Detection Store")]
  Worker -.->|upgrade event| WW["Warm Writer"]
```

The Motion Store keeps track of motion events. If there are several segments in sequence that have movements, they are considered a single motion event. We sample four frames from each event at 0/3, 1/3, 2/3 and 3/3. Using bounding boxes from the motion event, we crop the image to "zoom in" to the action, JPEG-encode the crops, and enqueue them as a job for a single global detection worker shared by all cameras. The worker is strictly serial — at most one in-flight Ollama request at any time — so a modest GPU is never hit with parallel load, and the analyzer never waits for the model: if the small queue is full the job is simply dropped with a warning (the motion event still records; only the object classification is lost).

The model is asked for structured output: the request carries a JSON schema (via Ollama's `format` field) with the configured class list as an enum, so the response is machine-parseable JSON with per-detection class, confidence, and a normalized bounding box. Responses are validated — out-of-range confidences, garbage boxes, and unknown classes are dropped. Every valid detection is recorded — there is no learned suppression that could silently drop a real detection. To stop a persistently busy area (a tree, a road) from generating events, paint it into the per-camera movement mask; to stop a stationary object (a parked car) from being classified without suppressing motion there, paint it into the per-camera detection mask, which blacks those cells out of every frame sent to the vision model.

We are still running in RAM, our 10 minute video buffer and some metadata stored in the motion and detection stores. At this stage we should have enough information to only save events that we care about on disk!

```mermaid
flowchart LR
  HotBuffer[("Hot Buffer (10m)")] -->|segments| MA["Motion Analyzer"]
  MS[("Motion Store")] --> MA
  DS[("Detection Store")] --> MA
  MA -->|event end| WarmWriter["Warm Writer"]
  WarmWriter --> DiskMovements
  WarmWriter --> DiskObjects
  WarmWriter --> Metadata
  WarmWriter --> Thumbnails

  subgraph Disc
    DiskMovements[("Movements")]
    DiskObjects[("Objects")]
    Metadata[("Metadata")]
    Thumbnails[("Thumbnails")]
  end

  Disc --> Client
  Prune -.-> Disc
```

The moment a motion run ends (post-padding elapsed), the analyzer assembles the complete event — pre-padding, motion, and post-padding pulled straight from the hot buffer, plus metadata from the motion and detection stores — and hands it to the warm writer, which persists it to disk immediately. This way an event only stays at risk in RAM for seconds after it ends, not until its segments age out of the buffer. The writer also saves metadata and thumbnails that is used by the web UI. User can stream saved video events via HLS.

Event writes never wait for the vision model. If an Ollama verdict arrives while the run is still open, it is picked up during assembly as before; if it arrives after the event is already on disk, the detection worker asks the warm writer to upgrade it post-hoc — the sidecar is rewritten with the detections and the files move from `movements/` to `objects/`, which switches the event to the longer object retention. All file mutations go through the warm writer, so writes and upgrades can never race. If a verdict lands in the tiny window while the event is being assembled, worst case the event simply stays movement-classified with the detections still visible in the detection store and API.

### Recording modes

The `storage` and `analytics` flags together select what gets recorded:

- **Event recording** (`storage.enabled = true`, `analytics.enabled = true`) — the default. Only motion and object events are saved, into `movements/` and `objects/`. Storage is proportional to activity.
- **Continuous recording** (`storage.enabled = true`, `analytics.enabled = false`) — "dumb NVR" mode. With no analyzer to gate on motion, a per-camera recorder rolls every segment from the hot buffer to disk in fixed-length chunks (each `max_event_duration_secs` long) under `continuous/`. Every chunk is a whole-GOP `.ts` that decodes on its own, and successive chunks carry a `"continues"` flag so the timeline stitches seamlessly. This is heavy — roughly **43 GB/day/camera at 4 Mbps** — so `continuous_retention_days` defaults to 1.
- **No recording** (`storage.enabled = false`) — live view only; nothing hits disk.

Both modes share the same warm writer, retention pruning, and HLS playback path; only the trigger differs.

### Durability

Event files are written atomically — staged as `.tmp`, fsynced, then renamed into place — so a crash or power cut never leaves a half-indexed event. On the next startup camon **recovers** interrupted writes instead of discarding them: an orphaned `.ts.tmp` (which may hold the footage of exactly the incident that cut the power) is trimmed to its last intact packet, its real duration recomputed from the stream timestamps, and indexed like any other event with a `"recovered": true` flag. If the disk runs low, a `min_free_bytes` guard emergency-prunes the oldest recordings (continuous first, then movements, then objects) so the writer keeps recording instead of failing.

### Storage backends

The warm event store sits behind a backend seam, so *where* events live is a config choice — analytics, motion settings, and the in-RAM hot buffer always stay local regardless.

- **Local disk** (default) — the `data_dir` on the machine running camon, with the atomic write ladder, crash recovery, and `min_free_bytes` free-space guard described above.
- **Remote stathost** — a [stathost](https://github.com/nsg/stathost) static file host, selected by adding a `[storage.stathost]` section. Each event becomes three sibling objects under `{camera}/` on the host — the `.ts` video, a `.json` sidecar (which here also carries the event **type**, since there are no `movements/`/`objects/`/`continuous/` directories), and eager `{stem}_thumb_{i}.jpg` filmstrip frames. Time-based retention still applies via `DELETE`; because the client can't see the server's disk, retention-by-space becomes a client-side **budget** (`max_stored_bytes`) that prunes the oldest events (continuous → movements → objects) when tracked usage exceeds it. Post-hoc movement→object upgrades just rewrite the sidecar in place — no object is ever renamed or moved.

  Use stathost **0.2.0 or later**: it has atomic uploads (readers never see a partially uploaded object), the detailed listing camon prefers for accurate storage accounting (`?detail=true`; camon falls back to the plain listing on older servers), and HTTP Range support. camon uploads the video first and the sidecar/thumbnails after, so an interruption at worst leaves a video with no sidecar (indexed as a plain movement event on the next scan). Warm-event playback **streams** rather than buffering the whole event in RAM: the playback handler forwards a single `Range` header to stathost and relays the `206`/`Content-Range` response, so seeking fetches only the requested bytes. Against an older server that ignores `Range` and answers `200`, the read degrades to streaming the full body.

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

Camon loads `config.toml` from the current working directory.

> **Note:** Pre-built binaries are linked against glibc. musl-based systems are not supported.

### Building from Source

Building needs nothing beyond a stable Rust toolchain (no OpenCV, no C++ toolchain — the vision code is pure Rust); ffmpeg is only needed at runtime:

```bash
sudo apt install ffmpeg
cargo build --release
./target/release/camon
```

## Configuration

Create a `config.toml` in the working directory. All sections are optional — defaults are shown below:

```toml
[update]
# Auto-update from GitHub Releases on startup (default: true).
# A downloaded binary is checked against the release's sha256sums.txt and
# rejected on mismatch (corruption protection, not a security guarantee), and
# must be a valid ELF before it replaces the running binary. Releases published
# without a sha256sums.txt are applied unverified with a warning.
enabled = true

[buffer]
# Hot buffer duration in seconds (default: 600 = 10 minutes)
hot_duration_secs = 600

[http]
# HTTP server port for web UI and API (default: 8080)
port = 8080

[analytics]
# Enable MOG2 motion detection (default: false)
enabled = true
# Frame sample rate for analysis (default: 5)
sample_fps = 5

# Object detection via Ollama (requires analytics enabled).
# One global worker serves all cameras, strictly serially (max one in-flight
# request); events never wait for the model.
[analytics.object_detection]
# Enable object detection on motion segments (default: false)
enabled = true
# Minimum confidence threshold (default: 0.5)
confidence_threshold = 0.5
# Object classes to detect (default: person, car, truck, dog, cat).
# Also constrains the model's structured JSON output.
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
# "continues"). In continuous mode this is the length of each chunk. 0 disables.
# Timing is monotonic, immune to camera PTS jumps.
max_event_duration_secs = 120
# Retention for movement-only events in days (default: 2)
movement_retention_days = 2
# Retention for object detection events in days (default: 14)
object_retention_days = 14
# Retention for continuous-recording chunks in days (default: 1). Short because
# continuous recording is ~43 GB/day/camera at 4 Mbps.
continuous_retention_days = 1
# Minimum free space (bytes) on the storage filesystem (default: 2 GiB).
# Below this, the oldest recordings are emergency-pruned (continuous →
# movements → objects) before each write. 0 disables the guard.
min_free_bytes = 2147483648

# Optional: send warm events to a remote stathost host instead of data_dir.
# See "Storage backends" above. When present and enabled, min_free_bytes is
# ignored in favour of the max_stored_bytes budget.
# [storage.stathost]
# url = "https://files.example.com"
# bucket = "camon"
# token = "s3cr3t-token"
# max_stored_bytes = 0

# Add one [[cameras]] block per camera
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
| `GET` | `/api/cameras/{id}/events?from=&to=` | Query warm events by time range |
| `GET` | `/api/cameras/{id}/events/{pts}/playlist.m3u8` | Warm event HLS playlist |
| `GET` | `/api/cameras/{id}/events/{pts}/segment` | Warm event HLS segment |
| `GET` | `/api/cameras/{id}/events/{pts}/thumbnail` | Warm event thumbnail JPEG |
| `GET` | `/api/cameras/{id}/events/{pts}/filmstrip/{index}` | Filmstrip frame JPEG |
| `GET` | `/api/cameras/{id}/detection-debug` | Detection debug entries |
| `GET` | `/api/cameras/{id}/detection-debug/{id}/frame/{index}` | Detection debug frame JPEG |
| `GET` | `/api/cameras/{id}/detection-debug/{id}/full-frame` | Detection debug full frame JPEG |

## Storage Tiers

| Tier | Medium | Retention | Quality | Purpose |
|---|---|---|---|---|
| Hot | RAM | ~10 minutes | 1080p @ 30fps | Live playback and real-time analysis |
| Warm | Disk | 2 days (movement) / 14 days (objects) / 1 day (continuous) | Original quality | Motion-triggered events, or gapless continuous recording when analytics is off |

## License

MIT — see [LICENSE.md](LICENSE.md).
