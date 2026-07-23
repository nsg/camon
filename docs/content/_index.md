+++
title = "Camon"
+++

## Storage Tiers

Video is stored across three tiers. Hot and warm tiers store data as-is from the camera (H.264 passthrough) for performance — no transcoding overhead. Hot storage keeps ~10 minutes in RAM at full quality (1080p @ 30fps) for live playback, scrollback, and real-time analysis while minimizing disk writes. Warm storage keeps motion events on disk for up to 2 days. Cold storage transcodes to lower resolution (480p @ 5fps) for long-term archival only (weeks–months).

Warm events are assembled from GOP-aligned segments (keyframe to keyframe) and written to disk the moment their motion run ends, with configurable pre-padding (default 5s) and post-padding (default 10s) to capture context around the event — an event is only at risk in RAM for seconds after it ends, not for the lifetime of the hot buffer. Typical GOP is 1-2 seconds (~750KB–1.5MB at 6 Mbps), though this depends on camera settings.

When a motion event ends, the system reaches back into the hot buffer for the preceding context (pre-padding before the first motion), capturing what led up to the event.

Access is abstracted behind a unified interface — consumers request video by time offset, and the system transparently serves from the appropriate tier.

## Camera Pipeline

Cameras stream H.264 via RTSP. The system uses FFmpeg to ingest RTSP streams, outputting MPEG-TS format to stdout. An MPEG-TS parser extracts H.264 frames and detects keyframes via the random_access_indicator in the adaptation field. Segments use wall-clock timestamps for timing.

## Camera Requirements

Cameras must provide an RTSP stream with H.264 codec at 1080p 30fps. Keyframe interval should be 1-2 seconds (GOP 30-60 frames) with bitrate around 6 Mbps. CBR or capped VBR recommended.

## Concurrency

Each camera has its own hot buffer using a single-producer, multi-consumer (SPMC) pattern. The ingestion thread writes while analytics and API read concurrently. Synchronization via `Arc<RwLock<HotBuffer>>` with minimal contention since there's only one writer per buffer.

## Analytics Pipeline

Cameras stream H.264 via RTSP into the hot buffer. Frames are sampled at 5fps for analysis. Motion detection (MOG2) produces scores and regions. Zones can be configured with different sensitivity — high sensitivity for doorways, normal for general areas, ignore for trees or busy roads. When motion is detected, object detection via Ollama vision LLM identifies objects from a configurable class list (default: person, car, truck, dog, cat). A 2x2 grid of frames from the motion event is sent to the model for classification. An optional fallback Ollama server can be configured for redundancy. Under heavy load, the pipeline gracefully degrades by reducing sample rate, then auto-recovers when load decreases.

## API

HTTP REST API for playback and search. Supports live and historical video playback by time offset, and event search by time range and camera. Authentication, semantic query, and clip export are planned.

## Web Interface

Vanilla HTML/CSS/JS served from the Rust binary with video playback via Vidstack (CDN). No build tools required — cargo builds everything. Provides live view with scrollback, timeline scrubbing across tiers (transparent to user), event search, and clip export.

## Error Handling

Camera disconnections are handled with automatic reconnection using exponential backoff (5s, doubling up to a 60s cap, with jitter); the delay resets after a stream stays healthy. A data watchdog reconnects a stream that stops delivering bytes, and a tripwire reconnects one that delivers data but produces no keyframes. Cameras operate independently — one disconnecting doesn't affect others.

## System Dependencies

Build requires OpenCV and Clang development headers. Runtime requires FFmpeg for RTSP ingestion, H.264 handling, and motion analysis frame decoding. On Ubuntu/Debian:

**Build:** `libopencv-dev`, `clang`, `libclang-dev`, `cmake`

**Runtime:** `ffmpeg`

Object detection requires a running [Ollama](https://ollama.com/) server with a vision model.

## Data Storage

Metadata is stored in memory. Video files are stored on disk.

Warm video files are stored per camera as `{data_dir}/{camera_id}/{movements|objects}/{timestamp}_{duration_ms}.ts`, where `data_dir` defaults to `/var/camon/storage`. Cold archives are organized by date as `cold/{year}/{month}/{event_id}_{timestamp}.mp4` (planned).
