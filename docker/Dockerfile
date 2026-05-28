# syntax=docker/dockerfile:1.7
#
# Reproducible dev image for agentic-architectures.
#
# Strategy:
#   - Python 3.11-slim base (3.11 = our floor + Playwright compatible)
#   - System deps for building wheels + Playwright Chromium prerequisites
#   - Install Python deps in a cached layer (just pyproject.toml + src/)
#   - Install Playwright Chromium with its OS deps
#   - Source code is mounted at runtime via docker-compose volume,
#     so edits on the host reflect inside the container live.
#
# Build standalone:
#   docker build -f docker/Dockerfile -t agentic-arch:dev .
#
# Run standalone (interactive shell, no host mount):
#   docker run -it --rm \
#     -e NEBIUS_API_KEY=$NEBIUS_API_KEY \
#     agentic-arch:dev bash

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Minimal system deps. Playwright's own installer will add browser-specific libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        build-essential \
        ca-certificates \
        bash-completion \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Layer 1: Python dependency install (cached unless pyproject changes)
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
RUN pip install --upgrade pip setuptools wheel \
 && pip install -e ".[dev,test,docs,nebius,faiss,tavily,networkx]"

# Layer 2: Playwright Chromium + OS deps (large layer, ~400MB)
RUN playwright install --with-deps chromium

# Layer 3: Register the Jupyter kernel papermill expects
RUN python -m ipykernel install --name python3 --display-name "Python 3"

# Layer 4: Pre-load the rest of the source (overridden at runtime by volume mounts)
COPY notebooks/ ./notebooks/
COPY tests/ ./tests/
COPY benchmarks/ ./benchmarks/
COPY docs/ ./docs/
COPY scripts/ ./scripts/
COPY mkdocs.yml .env.example ./

# Useful default for one-off `docker run`
CMD ["bash"]
