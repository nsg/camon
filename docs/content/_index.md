+++
title = "Camon"
+++

## Storage Tiers

Video is stored across three tiers. Hot and warm tiers store data as-is from the camera (H.264 passthrough) for performance — no transcoding overhead. Hot storage keeps ~10 minutes in RAM at full quality (1080p @ 30fps) for live playback, scrollback, and real-time analysis while minimizing disk writes. Warm storage flushes footage to disk for up to 2 days. Cold storage transcodes to lower resolution (480p @ 5fps) for long-term archival only (weeks–months).

Hot flushes to warm in GOP-aligned segments (keyframe to keyframe) with 16KB alignment for SSD efficiency. Only segments with motion (detected by MOG2) are written — static footage is discarded, never hitting disk. Typical GOP is 1-2 seconds (~750KB–1.5MB at 6 Mbps), though this depends on camera settings.

When object detection identifies a person or vehicle, the system reaches back into the hot buffer to save preceding context (e.g., T-30s to T+10s), capturing what led up to the event.

Access is abstracted behind a unified interface — consumers request video by time offset, and the system transparently serves from the appropriate tier.

## Camera Pipeline

Cameras stream H.264 via RTSP. The system uses FFmpeg to ingest RTSP streams, outputting MPEG-TS format to stdout. An MPEG-TS parser extracts H.264 frames and detects keyframes via the random_access_indicator in the adaptation field. PTS timestamps are extracted from PES headers for accurate timing.

## Camera Requirements

Cameras must provide an RTSP stream with H.264 codec at 1080p 30fps. Keyframe interval should be 1-2 seconds (GOP 30-60 frames) with bitrate around 6 Mbps. CBR or capped VBR recommended.

## Concurrency

Each camera has its own hot buffer using a single-producer, multi-consumer (SPMC) pattern. The ingestion thread writes while analytics and API read concurrently. Synchronization via `Arc<RwLock<HotBuffer>>` with minimal contention since there's only one writer per buffer.

## Analytics Pipeline

Cameras stream H.264 via RTSP into the hot buffer. Frames are sampled at 5fps for analysis. Motion detection (MOG2) produces scores and regions. Zones can be configured with different sensitivity — high sensitivity for doorways, normal for general areas, ignore for trees or busy roads. When motion is detected, object detection (YOLO on CPU) identifies people, vehicles, and animals. Object detection crops to motion regions rather than processing the full frame, improving small object detection ~3x. LLM analysis runs on warm tier events via Ollama, generating summaries and embeddings for semantic search. Under heavy load, the pipeline gracefully degrades by reducing sample rate, then auto-recovers when load decreases.

## API

HTTP REST API for playback and search with bearer token authentication. Supports live and historical video playback by time offset, event search by time range, camera, or semantic query, and clip export.

## Web Interface

Vanilla HTML/CSS/JS served from the Rust binary with video playback via Vidstack (CDN). No build tools required — cargo builds everything. Provides live view with scrollback, timeline scrubbing across tiers (transparent to user), event search, and clip export.

## Error Handling

Camera disconnections are handled with automatic reconnection after a 5-second delay. Cameras operate independently — one disconnecting doesn't affect others. Reconnected cameras resume immediately.

## System Dependencies

Build requires OpenCV development headers. Runtime requires FFmpeg for RTSP ingestion and H.264 handling. On Ubuntu/Debian:

**Build:** `libopencv-dev`

**Runtime:** `ffmpeg`

SQLite is bundled by rusqlite. ONNX Runtime is auto-downloaded by the ort crate.

## Data Storage

Metadata is stored in SQLite (WAL mode). Video files are stored on disk with references in the database. Semantic search is enabled via vector embeddings using sqlite-vec.

All data lives under `/var/camon/`. Warm video files are stored per camera as `warm/{camera_id}/{timestamp}_{duration_ms}.h264`. Cold archives are organized by date as `cold/{year}/{month}/{event_id}_{timestamp}.mp4`. The database is at `db/camon.db`.
