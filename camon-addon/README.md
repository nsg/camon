# Camon

Multi-camera NVR with real-time motion and object analytics. Streams RTSP
cameras into a 10-minute RAM buffer, records only motion/object events, and
classifies them with an Ollama vision model — with the web UI embedded in the
Home Assistant sidebar via ingress.

Configured by a `camon.toml` file (the same format as a native Camon install) —
see the Documentation tab. amd64 only.
