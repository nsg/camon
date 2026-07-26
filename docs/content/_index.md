+++
title = "Camon"
+++

## Storage Tiers

Video is stored across three tiers. Hot and warm tiers store data as-is from the camera (H.264 passthrough) for performance — no transcoding overhead. Hot storage keeps ~10 minutes in RAM at full quality (1080p @ 30fps) for live playback, scrollback, and real-time analysis while minimizing disk writes. Warm storage keeps motion events on disk for up to 2 days. Cold storage transcodes to lower resolution (480p @ 5fps) for long-term archival only (weeks–months).

Warm events are assembled from GOP-aligned segments (keyframe to keyframe) and written to disk the moment their motion run ends, with configurable pre-padding (default 5s) and post-padding (default 10s) to capture context around the event — an event is only at risk in RAM for seconds after it ends, not for the lifetime of the hot buffer. Typical GOP is 1-2 seconds (~750KB–1.5MB at 6 Mbps), though this depends on camera settings.

A single event is capped at `max_event_duration_secs` (default 120s). Sustained motion past the cap is flushed as a complete, independently playable chunk and continues in a new one, keeping RAM and file sizes bounded and preventing gaps when a run outlives the hot buffer. Follow-on chunks carry no pre-padding and a `"continues"` sidecar flag so the UI can stitch the chain. All event-lifecycle timing — the post-padding countdown and the duration cap — runs on a monotonic clock rather than media PTS, so a camera timestamp jump or reset can't stall or prematurely close an event; media timing (segment durations, playlist math, filenames) stays on PTS.

When a motion event ends, the system reaches back into the hot buffer for the preceding context (pre-padding before the first motion), capturing what led up to the event.

### Recording modes

What warm storage records depends on the `storage` and `analytics` flags together. With analytics enabled it is **event recording**: only motion and object events land on disk, in `movements/` and `objects/`. With analytics disabled the same combination becomes **continuous recording** — a "dumb NVR" mode. Without an analyzer to gate on motion, a per-camera recorder rolls every hot-buffer segment to disk in fixed-length chunks (each `max_event_duration_secs` long) under `continuous/`, flagging successive chunks `"continues"` so the timeline stitches together. Because chunks split on whole GOP segments, each `.ts` starts with PAT/PMT and a keyframe and plays on its own. Continuous recording is heavy — roughly 43 GB/day/camera at 4 Mbps — so `continuous_retention_days` defaults to 1, far shorter than event retention. Both modes share the same warm writer, retention pruning, and HLS playback; only the trigger differs. Disabling storage entirely leaves live view only.

Access is abstracted behind a unified interface — consumers request video by time offset, and the system transparently serves from the appropriate tier.

## Camera Pipeline

Cameras stream H.264 via RTSP. The system uses FFmpeg to ingest RTSP streams, outputting MPEG-TS format to stdout. An MPEG-TS parser extracts H.264 frames and detects keyframes via the random_access_indicator in the adaptation field. Segments use wall-clock timestamps for timing.

## Camera Requirements

Cameras must provide an RTSP stream with H.264 codec at 1080p 30fps. Keyframe interval should be 1-2 seconds (GOP 30-60 frames) with bitrate around 6 Mbps. CBR or capped VBR recommended.

## Concurrency

Each camera has its own hot buffer using a single-producer, multi-consumer (SPMC) pattern. The ingestion thread writes while analytics and API read concurrently. Synchronization via `Arc<RwLock<HotBuffer>>` with minimal contention since there's only one writer per buffer.

## Analytics Pipeline

Cameras stream H.264 via RTSP into the hot buffer. Keyframes are sampled for analysis — with a 1-second GOP that is one frame per second. Motion detection (a built-in pure-Rust MOG2 background subtractor with a ~5 minute background memory, followed by morphological opening and connected-component filtering) produces scores and regions. Detection is deterministic and controlled per camera by three settings you edit live in the web UI: a sensitivity slider (MOG2 var_threshold), a minimum-object-size slider, and a paintable movement mask (a 16×12 grid — mask trees or busy roads so motion there is excluded). There is no automatic tuning. A second painted layer on the same grid, the detection mask, is independent of motion detection: its cells are blacked out of every frame sent to the vision model, so a stationary nuisance object (say a parked car) can be hidden from classification without suppressing motion there. When motion is detected, object detection via Ollama vision LLM identifies objects from a configurable class list (default: person, car, truck, dog, cat). Up to four cropped frames from the motion event are queued as a job for a single global detection worker shared by all cameras. The worker is strictly serial — at most one in-flight Ollama request at any time — so a modest GPU never sees parallel load, and motion detection never waits for the model (a full queue drops the job; the motion event still records). Requests use Ollama's structured output with a JSON schema — the class list as an enum, numeric confidence and normalized bounding box — and responses are validated before use; garbage is dropped. Every valid detection is recorded — nothing is silently suppressed. Events are written to disk the moment their motion run ends; a verdict that arrives later upgrades the on-disk event post-hoc from a movement to an object event (sidecar rewritten, files moved, retention class switched). An optional fallback Ollama server can be configured for redundancy.

## API

HTTP REST API for playback and search. Supports live and historical video playback by time offset, and event search by time range and camera. Authentication, semantic query, and clip export are planned.

## Web Interface

Vanilla HTML/CSS/JS served from the Rust binary with video playback via Vidstack (CDN). No build tools required — cargo builds everything. Provides live view with scrollback, timeline scrubbing across tiers (transparent to user), event search, and clip export.

## Home Assistant

Beyond running as an ingress add-on, camon can bridge to Home Assistant over MQTT (`[mqtt]` in the config; auto-configured from the Supervisor in the add-on). It publishes retained MQTT discovery messages so each camera materializes as a native HA device with three entity kinds: a snapshot camera fed with JPEG frames only while motion is active (decoded on demand from the newest hot-buffer segment — idle cameras cost nothing), a motion binary sensor tracking the physical motion run, and one occupancy binary sensor per object-detection class that holds for a configurable time after the last sighting. The bridge is strictly outbound — it subscribes to nothing — and marks all entities unavailable via a broker last-will if camon dies.

## Error Handling

Camera disconnections are handled with automatic reconnection using exponential backoff (5s, doubling up to a 60s cap, with jitter); the delay resets after a stream stays healthy. A data watchdog reconnects a stream that stops delivering bytes, and a tripwire reconnects one that delivers data but produces no keyframes. The analysis decoders get the same treatment: handing a segment to a decoder is bounded rather than blocking, so an FFmpeg process that stops reading its input is killed and respawned instead of freezing motion analysis, and a decoder that consumes segments without emitting frames is caught by a zero-frame tripwire and respawned too. Cameras operate independently — one disconnecting doesn't affect others.

## System Dependencies

Camon builds into a single self-contained binary with a plain stable Rust toolchain — no OpenCV, no C++ toolchain, no native vision libraries. The motion detector (MOG2, morphology, connected components) and all image handling (cropping, JPEG encoding via the pure-Rust `image` crate) are Rust. Runtime requires FFmpeg for RTSP ingestion, H.264 handling, and motion analysis frame decoding. On Ubuntu/Debian:

**Build:** stable Rust only

**Runtime:** `ffmpeg`

Object detection requires a running [Ollama](https://ollama.com/) server with a vision model.

## Data Storage

Metadata is stored in memory. Video files are stored on disk.

Warm video files are stored per camera as `{data_dir}/{camera_id}/{movements|objects|continuous}/{timestamp}_{duration_ms}.ts`, where `data_dir` defaults to `/var/camon/storage`. The `continuous/` subdirectory holds chunks written in continuous-recording mode; `movements/` and `objects/` hold motion and object events. Cold archives are organized by date as `cold/{year}/{month}/{event_id}_{timestamp}.mp4` (planned).

Writes are durable: each event is staged as a `.tmp` file, fsynced, and renamed into place, so a crash or power cut never yields a torn or half-indexed event. At the next startup any interrupted `.ts.tmp` is recovered rather than deleted — trimmed to the last intact MPEG-TS packet, its duration recomputed from the stream's PES timestamps, and indexed with a `"recovered": true` sidecar flag — because an interrupted write may hold the footage of the very incident that cut the power. A `min_free_bytes` guard (default 2 GiB) emergency-prunes the oldest recordings — continuous first, then movements, then objects — whenever the storage filesystem runs low, so recording continues instead of failing on a full disk.

The warm store sits behind a backend seam, so it need not be local. The default is local disk (above); adding a `[storage.stathost]` section instead sends warm events to a remote [stathost](https://github.com/nsg/stathost) static file host over HTTP, while analytics, motion settings, and the hot buffer stay local. On the host each event is three sibling objects under `{camera_id}/` — `{timestamp}_{duration_ms}.ts`, a `.json` sidecar, and eager `{stem}_thumb_{i}.jpg` filmstrip frames — with no directory tiers, so the sidecar carries the event **type** as well as its detections, and a movement→object upgrade simply rewrites that sidecar in place. Time-based retention still deletes expired events; retention-by-space becomes a client-side budget (`max_stored_bytes`), since the client can't see the server's disk. Because stathost has no atomic PUT yet, the video is uploaded before the sidecar, so an interrupted upload degrades to a video indexed as a plain movement event rather than a lost one.
