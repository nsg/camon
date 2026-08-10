# Camon Home Assistant add-on image.
#
# Build approach: compile Camon from the workflow's checked-out source, then
# run it on a slim Debian runtime with ffmpeg installed (Camon spawns
# ffmpeg/ffprobe). Release images are built only on v* pushes, so the complete
# image recipe and its application source come from the same release tag.
#
# We deliberately do NOT use a Home Assistant base image here. The HA base
# images are Alpine/musl, whereas the Rust toolchain image below is glibc; a
# glibc binary won't run on musl. Debian bookworm on both the build and run
# stages keeps them ABI-compatible and gives us a well-supported ffmpeg. The
# catalog manifest in nsg/ha-repo sets `init: false`, so our CMD runs as PID 1
# and receives SIGTERM directly (Camon shuts down gracefully on SIGTERM). The
# same manifest limits the image to amd64.
#
# This Dockerfile is built by GitHub Actions (.github/workflows/addon.yml) and
# pushed to ghcr.io/nsg/amd64-addon-camon. The catalog manifest in
# https://github.com/nsg/ha-repo points Home Assistant at that prebuilt image.

# ---- Build stage -----------------------------------------------------------
FROM rust:1-bookworm AS builder
WORKDIR /src
ARG CAMON_BUILD_VERSION=dev
COPY Cargo.toml Cargo.lock build.rs config.toml.example ./
COPY src ./src
# The workflow derives CAMON_BUILD_VERSION from the checkout and validates a
# release value against Cargo.toml before this build starts.
RUN CAMON_BUILD_VERSION="${CAMON_BUILD_VERSION}" cargo build --release --locked --bin camon

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
# The commented config template from the checked-out source — run.sh seeds it
# into the add-on's config folder on first start.
COPY --from=builder /src/config.toml.example /usr/local/share/camon/config.toml.example
COPY camon-addon/run.sh /run.sh
RUN chmod a+x /run.sh

LABEL \
    org.opencontainers.image.title="Camon" \
    org.opencontainers.image.description="Multi-camera NVR with real-time analytics and tiered storage" \
    org.opencontainers.image.source="https://github.com/nsg/camon" \
    org.opencontainers.image.licenses="MIT"

CMD [ "/run.sh" ]
