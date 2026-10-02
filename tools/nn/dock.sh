#!/bin/bash
# Runs a Python module of this repo in the training container (PyTorch + CUDA):
#   bash tools/nn/dock.sh tools.nn.gamedata
# The image is snake-ai-trainer (PyTorch 2.13, CUDA); the repo is mounted at /repo.
# DOCK_NAME=t2-test5 bash tools/nn/dock.sh ...   names the container (so it can be told apart and stopped).
# torch.compile keeps its kernels between runs in the Docker volume tww3-torch-cache: a run that
# compiles what an earlier one did loads it (the warm-up of training and evaluation gets shorter).
repo="$(cd "$(dirname "$0")/../.." && pwd -W 2>/dev/null || pwd)"
MSYS_NO_PATHCONV=1 exec docker run --rm ${DOCK_NAME:+--name "$DOCK_NAME"} --gpus all --shm-size 2g -v "${repo}:/repo" -w /repo -e PYTHONPATH=/repo \
  -v tww3-torch-cache:/cache -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor -e TRITON_CACHE_DIR=/cache/triton \
  -e PYTHONUNBUFFERED=1 snake-ai-trainer python -m "$@"
