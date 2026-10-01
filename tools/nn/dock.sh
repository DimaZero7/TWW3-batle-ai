#!/bin/bash
# Runs a Python module of this repo in the training container (PyTorch + CUDA):
#   bash tools/nn/dock.sh tools.nn.gamedata
# The image is snake-ai-trainer (PyTorch 2.13, CUDA); the repo is mounted at /repo.
# DOCK_NAME=t2-test5 bash tools/nn/dock.sh ...   names the container (so it can be told apart and stopped).
repo="$(cd "$(dirname "$0")/../.." && pwd -W 2>/dev/null || pwd)"
MSYS_NO_PATHCONV=1 exec docker run --rm ${DOCK_NAME:+--name "$DOCK_NAME"} --gpus all --shm-size 2g -v "${repo}:/repo" -w /repo -e PYTHONPATH=/repo \
  -e PYTHONUNBUFFERED=1 snake-ai-trainer python -m "$@"
