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

```
IP Camera ──RTSP──▶ Camon
                    │
                    ▼
              FFmpeg ──▶ H.264 frames
                    │
            ┌───────┴───────┐
            ▼               ▼
      ┌──────────┐   ┌───────────┐
      │Hot Buffer│   │ Analytics │
      │(RAM ~10m)│   │   @5fps   │
      └────┬─────┘   │           │
           │         │ MOG2 ──▶ motion
           │         │ Ollama ──▶ detections
           │         └───────────┘
           ▼
    ┌──────────────┐
    │ Warm Storage │◀── on motion
    │   (disk)     │
    └──────────────┘

    Axum HTTP ──▶ HLS + REST + UI
```

## Features

- **RTSP ingestion** — H.264 streams from IP cameras via FFmpeg
- **Motion detection** — MOG2 background subtraction with adaptive percentile-based thresholding
- **Object detection** — vision LLM inference via Ollama with fallback server support
- **Tiered storage** — hot (RAM) for live playback, warm (disk) for event recordings
- **HLS streaming** — live and recorded event playback over HTTP
- **Web UI** — live monitor, event browser, and event playback as separate focused views
- **REST API** — query events by time range, camera, and type
- **Auto-update** — checks GitHub Releases on startup for new versions

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
| `GET` | `/api/cameras/{id}/motion/background` | JPEG learned background model |
| `GET` | `/api/cameras/{id}/detection/grid` | Detection heatmap grid (JSON) |
| `GET` | `/api/cameras/{id}/detections` | Detected objects with confidence |
| `GET` | `/api/cameras/{id}/detections/{id}/frame` | JPEG frame of detection |
| `GET` | `/api/cameras/{id}/events?from=&to=` | Query warm events by time range |
| `GET` | `/api/cameras/{id}/events/{pts}/playlist.m3u8` | Warm event HLS playlist |
| `GET` | `/api/cameras/{id}/events/{pts}/segment` | Warm event HLS segment |
| `GET` | `/api/cameras/{id}/events/{pts}/thumbnail` | Warm event thumbnail JPEG |

## Storage Tiers

| Tier | Medium | Retention | Quality | Purpose |
|---|---|---|---|---|
| Hot | RAM | ~10 minutes | 1080p @ 30fps | Live playback and real-time analysis |
| Warm | Disk | 2 days (movement) / 14 days (objects) | Original quality | Motion-triggered event recordings |

## License

MIT — see [LICENSE.md](LICENSE.md).
