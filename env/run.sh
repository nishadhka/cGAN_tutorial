#!/usr/bin/env bash
# Build the image once, then `./run.sh` to start it.
# Override the command at the end, e.g.  `./run.sh bash`  or  `./run.sh pytf main.py`.
set -euo pipefail

IMAGE="${IMAGE:-cgan-gpu:latest}"
HOST_MOUNT="${HOST_MOUNT:-/scratch/notebook}"
PORT="${PORT:-8888}"

# Build only if the image is missing
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo ">>> Building $IMAGE"
  docker build -t "$IMAGE" "$(dirname "$0")"
fi

exec docker run --gpus all --rm -it \
  --shm-size=8g --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p "${PORT}:8888" \
  -v "${HOST_MOUNT}:/workspace" \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  "$IMAGE" "$@"
