<div align="center">
  <h1>Camon</h1>
  <p>Multi-camera video surveillance with real-time analytics and tiered storage.</p>
</div>

---

## About

Camon is a self-hosted video surveillance system that ingests RTSP streams from IP cameras, runs real-time motion and object detection, and manages footage across three storage tiers. It serves a web UI and REST API for live viewing, timeline scrubbing, event search, and clip export — all from a single binary.

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
           │         │ MOG2 ──▶ events
           │         │ YOLO ──▶ detections
           │         └───────────┘
           ▼
    ┌──────────────┐
    │ Warm Storage │◀── on motion
    │ (disk ~2d)   │
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │ Cold Archive │
    │ (transcoded) │
    └──────────────┘

    Axum HTTP ──▶ HLS + REST + UI
```

## Features

- **RTSP ingestion** — H.264 streams from IP cameras via FFmpeg
- **Motion detection** — MOG2 background subtraction with adaptive percentile-based thresholding
- **Object detection** — YOLO26n inference on CPU via ONNX Runtime
- **Tiered storage** — hot (RAM), warm (disk), and cold (transcoded archive)
- **HLS streaming** — live and historical playback over HTTP
- **REST API** — event search by time range, camera, motion intensity, or detected objects
- **Web UI** — live view, timeline scrubbing, event search, and clip export


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

Create a `config.toml` to configure cameras, storage, and analytics. See `config.toml.example` for a fully commented reference.

| Section | Key | Default | Description |
|---|---|---|---|
| `[buffer]` | `hot_duration_secs` | `600` | Duration of in-memory hot buffer (seconds) |
| `[http]` | `port` | `8080` | HTTP server port |
| `[analytics]` | `enabled` | `false` | Enable motion detection pipeline |
| `[analytics]` | `sample_fps` | `5` | Frame sampling rate for analysis |
| `[analytics.object_detection]` | `enabled` | `false` | Enable YOLO object detection |
| `[analytics.object_detection]` | `model_path` | HuggingFace URL | Path or URL to YOLO26n ONNX model |
| `[analytics.object_detection]` | `confidence_threshold` | `0.5` | Minimum detection confidence |
| `[analytics.object_detection]` | `classes` | `["person", "car", ...]` | Object classes to detect |
| `[storage]` | `enabled` | `true` | Enable warm disk storage |
| `[storage]` | `data_dir` | `/var/camon/storage` | Storage directory path |
| `[storage]` | `pre_padding_secs` | `5` | Seconds of video before motion event |
| `[storage]` | `post_padding_secs` | `10` | Seconds of video after motion event |

Cameras are defined as TOML array entries:

```toml
[[cameras]]
id = "front-door"
url = "rtsp://user:pass@192.168.1.100:554/stream1"
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
| `GET` | `/api/cameras/{id}/motion` | Motion events with timestamps |
| `GET` | `/api/cameras/{id}/motion/{seq}/mask` | JPEG motion mask overlay |
| `GET` | `/api/cameras/{id}/detections` | Detected objects with confidence |
| `GET` | `/api/cameras/{id}/detections/{id}/frame` | JPEG frame of detection |
| `GET` | `/api/cameras/{id}/events?from=&to=` | Query events by time range |
| `GET` | `/api/cameras/{id}/events/{pts}/playlist.m3u8` | Warm event HLS playlist |
| `GET` | `/api/cameras/{id}/events/{pts}/segment` | Warm event segment |

## Storage Tiers

| Tier | Medium | Retention | Quality | Purpose |
|---|---|---|---|---|
| Hot | RAM | ~10 minutes | 1080p @ 30fps | Live playback and analysis |
| Warm | Disk | Up to 2 days | Original quality | Motion-triggered event segments |
| Cold | Disk | Weeks–months | 480p @ 5fps | Long-term transcoded archive |

## License

MIT — see [LICENSE.md](LICENSE.md).
