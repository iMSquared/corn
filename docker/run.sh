#!/usr/bin/env bash

set -exu

# CONFIGURE DATA PATHS
IG_PATH="/home/corn/isaacgym"
CACHE_PATH="/home/corn/.cache/pkm"
DATA_PATH="/home/corn/datasets/"

# Figure out repository root.
SCRIPT_DIR="$( cd "$( dirname $(realpath "${BASH_SOURCE[0]}") )" && pwd )"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"

# Create a temporary directory to be shared between host<->docker.
mkdir -p "${CACHE_PATH}"
mkdir -p '/tmp/docker/'

# Launch docker with the following configuration:
# * Display/Gui connected
# * Network enabled (passthrough to host)
# * Privileged
# * GPU devices visible
# * Current working git repository mounted at ${HOME}
# * 8Gb Shared Memory
# NOTE: comment out `--network host` for profiling with `nsys-ui`.
docker run -it \
    --mount type=bind,source="${REPO_ROOT}",target="/home/user/$(basename ${REPO_ROOT})" \
    --mount type=bind,source="${IG_PATH}",target="/opt/isaacgym/" \
    --mount type=bind,source="${CACHE_PATH}",target="/home/user/.cache/pkm" \
    --mount type=bind,source="${DATA_PATH}",target="/input" \
    --mount type=bind,source=/tmp/docker,target="/tmp/docker" \
    --shm-size=32g \
    --network host \
    --privileged \
    --gpus all \
    "$@" \
    "pkm:latest"
