# Camon Home Assistant add-on image.
#
# Build approach: compile Camon from source (keeps the add-on self-contained —
# no dependency on a published GitHub release binary), then run it on a slim
# Debian runtime with ffmpeg installed (Camon spawns ffmpeg/ffprobe).
#
# We deliberately do NOT use a Home Assistant base image here. The HA base
# images are Alpine/musl, whereas the Rust toolchain image below is glibc; a
# glibc binary won't run on musl. Debian bookworm on both the build and run
# stages keeps them ABI-compatible and gives us a well-supported ffmpeg. With
# `init: false` in config.yaml our CMD runs as PID 1 and receives SIGTERM
# directly (Camon shuts down gracefully on SIGTERM). amd64 only — see
# config.yaml `arch`.
#
# This Dockerfile is built by GitHub Actions (.github/workflows/addon.yml) and
# pushed to ghcr.io/nsg/amd64-addon-camon; the Home Assistant supervisor pulls
# that prebuilt image (config.yaml `image`) rather than building on-device.

# ---- Build stage -----------------------------------------------------------
FROM rust:1-bookworm AS builder
# The image is built with THIS DIRECTORY (camon-addon/) as the Docker
# context, so the Rust source is not in the context at all. Clone the
# repository at the version tag matching config.yaml instead (the workflow
# passes CAMON_VERSION derived from config.yaml) — pinned, so the add-on
# always builds the exact released source.
ARG CAMON_VERSION=v0.4.0
RUN git clone --depth 1 --branch ${CAMON_VERSION} https://github.com/nsg/camon /src
WORKDIR /src
RUN cargo build --release --locked --bin camon

# ---- Runtime stage ---------------------------------------------------------
FROM debian:bookworm-slim

# ffmpeg/ffprobe are spawned by Camon; ca-certificates for HTTPS (Ollama,
# stathost, update checks); curl + jq for run.sh to query and parse the
# Supervisor's /services/mqtt response (MQTT auto-configuration) — the slim
# base image ships neither.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        ca-certificates \
        curl \
        jq \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/target/release/camon /usr/local/bin/camon
# The commented config template from the cloned source — run.sh seeds this into
# the add-on's config folder (/config/camon.toml.example) on first start.
COPY --from=builder /src/config.toml.example /usr/local/share/camon/config.toml.example
# Context is camon-addon/, so run.sh sits at the context root.
COPY run.sh /run.sh
RUN chmod a+x /run.sh

LABEL \
    org.opencontainers.image.title="Camon" \
    org.opencontainers.image.description="Multi-camera NVR with real-time analytics and tiered storage" \
    org.opencontainers.image.source="https://github.com/nsg/camon" \
    org.opencontainers.image.licenses="MIT"

CMD [ "/run.sh" ]
