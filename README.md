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

Each camera runs its own independent pipeline with its own buffer. From the buffer we feed clients (browsers watching the live stream) and the motion analyzer. As segments age out of the buffer, those marked with motion are written to disk.

```mermaid
flowchart LR
  HotBuffer[("Hot Buffer (10m)")] -->|HLS| Clients
  HotBuffer -.->|evicted| WW[("Warm Writer (disk)")]
  HotBuffer -->|keyframes| MA["Motion Analyzer"] --> MS[("Motion Store")]
  MA -->|Ollama| OD["Object Detection"] --> DS[("Detection Store")]
  WW -.- MS
  WW -.- DS
```

Clients can stream the raw segments from the buffer directly, the only thing we need to do is to serve a playlist.m3u8 which is a simple text file to the player. Object Detection is heavy on the CPU (or GPU) so most "detections" are filtered in the Motion Analyzer stage.

```mermaid
flowchart LR
  HotBuffer[("Hot Buffer (10m)")] -->|keyframes| Decode["Decode 320x240 (grayscale)"] --> MA
  MA -.-> AutomaticTuning("Automatic Tuning") -.-> MA
  MA --> MS[("Motion Store")]

subgraph MA["Motion Analyzer"]
   MOG2["Background Subtraction"] --> Shadow["Shadow Removal"] --> Morph["Morphological Opening"] --> Contour["Contour Filtering"]
end
```

To make decoding as lightweight as possible, we only extract keyframes because they are self-contained and do not depend on surrounding frames. This also has the added bonus of reducing the framerate.

The Motion Analyzer uses OpenCV's MOG2 (Mixture of Gaussians) background subtractor to detect foreground motion, followed by shadow removal, morphological opening to eliminate noise, and contour filtering to discard small blobs. Several of these parameters are tuned automatically based on the camera's noise level to find an equilibrium between sensitivity and noise suppression.

```mermaid
flowchart LR
  HotBuffer[("Hot Buffer (10m)")] -->|motion event| Sub["Subsample 4 Frames"]
  MA[("Motion Store")] -->|bounding boxes| Sub
  Sub --> Crop1["Crop"] --> Ollama1["Ollama"] --> Grid["Detection Grid (16x12)"]
  Sub --> Crop2["Crop"] --> Ollama2["Ollama"] --> Grid
  Sub --> Crop3["Crop"] --> Ollama3["Ollama"] --> Grid
  Sub --> Crop4["Crop"] --> Ollama4["Ollama"] --> Grid
  Grid --> DS[("Detection Store")]
```

The Motion Store keeps track of motion events. If there are several segments in sequence that have movements, they are considered a single motion event. We sample four frames from each event at 0/3, 1/3, 2/3 and 3/3. Using bounding boxes from the motion event, we crop the image to "zoom in" to the action before sending it to our vision model running in Ollama. We keep track of where in the frame the object was detected and record it in a detection grid. If there are many detections of a specific class of objects in that area, the detection is suppressed (dropped). For example, this prevents repeated detections for a parked car.

We are still running in RAM, our 10 minute video buffer and some metadata stored in the motion and detection stores. At this stage we should have enough information to only save events that we care about on disk!

```mermaid
flowchart LR
  HotBuffer[("Hot Buffer (10m)")] -.->|evicted| WarmWriter["Warm Writer"]
  MS[("Motion Store")] --> WarmWriter
  DS[("Detection Store")] --> WarmWriter
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

As segments age out of the hot buffer, the warm writer checks the motion and detection stores to decide what to keep. Events are written to disk with padding before and after motion for context. The writer also saves metadata and thumbnails that is used by the web UI. User can stream saved video events via HLS.

## Quick Start

Install FFmpeg and download the latest binary from [GitHub Releases](https://github.com/nsg/camon/releases):

```bash
# Ubuntu 24.10+
sudo apt install ffmpeg libopencv-contrib406t64

# Other Ubuntu/Debian (pulls in extra -dev files)
sudo apt install ffmpeg libopencv-dev

curl -fLO https://github.com/nsg/camon/releases/latest/download/camon-linux-glibc
chmod +x camon-linux-glibc
./camon-linux-glibc
```

Camon loads `config.toml` from the current working directory.

> **Note:** Pre-built binaries are linked against glibc. musl-based systems are not supported.

### Building from Source

Install system dependencies:

```bash
sudo apt install libopencv-dev clang libclang-dev cmake ffmpeg
```

```bash
cargo build --release
./target/release/camon
```

## Configuration

Create a `config.toml` in the working directory. All sections are optional — defaults are shown below:

```toml
[update]
# Auto-update from GitHub Releases on startup (default: true)
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

# Object detection via Ollama (requires analytics enabled)
[analytics.object_detection]
# Enable object detection on motion segments (default: false)
enabled = true
# Minimum confidence threshold (default: 0.5)
confidence_threshold = 0.5
# Object classes to detect (default: person, car, truck, dog, cat)
classes = ["person", "car", "truck", "dog", "cat"]

[analytics.object_detection.ollama]
# Ollama server URL (default: http://localhost:11434)
url = "http://localhost:11434"
# Vision model to use (default: gemma4:e4b)
model = "gemma4:e4b"

# Optional fallback server if primary fails
# [analytics.object_detection.ollama.fallback]
# url = "http://backup:11434"
# model = "gemma3:4b"

[storage]
# Enable warm storage — flush motion events to disk (default: true)
enabled = true
# Directory for event files (default: /var/camon/storage)
data_dir = "/var/camon/storage"
# Seconds of context before first motion in an event (default: 5)
pre_padding_secs = 5
# Seconds of context after last motion in an event (default: 10)
post_padding_secs = 10
# Retention for movement-only events in days (default: 2)
movement_retention_days = 2
# Retention for object detection events in days (default: 14)
object_retention_days = 14

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
| `GET` | `/api/cameras/{id}/motion/stability` | JPEG motion foreground mask |
| `GET` | `/api/cameras/{id}/motion/stability/raw` | JPEG raw MOG2 output (with shadows) |
| `GET` | `/api/cameras/{id}/motion/stability/no-shadow` | JPEG after shadow removal |
| `GET` | `/api/cameras/{id}/motion/stability/morph` | JPEG after morphological opening |
| `GET` | `/api/cameras/{id}/motion/background` | JPEG learned background model |
| `GET` | `/api/cameras/{id}/motion/tuner` | Adaptive tuner stats (JSON) |
| `GET` | `/api/cameras/{id}/detection/grid` | Detection heatmap grid (JSON) |
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

## Storage Tiers

| Tier | Medium | Retention | Quality | Purpose |
|---|---|---|---|---|
| Hot | RAM | ~10 minutes | 1080p @ 30fps | Live playback and real-time analysis |
| Warm | Disk | 2 days (movement) / 14 days (objects) | Original quality | Motion-triggered event recordings |

## License

MIT — see [LICENSE.md](LICENSE.md).
