#!/bin/bash
# Runs a Python module of this repo in the training container (PyTorch + CUDA):
#   bash tools/nn/dock.sh tools.nn.gamedata
# The image is snake-ai-trainer (PyTorch 2.13, CUDA); the repo is mounted at /repo.
# DOCK_NAME=t2-test5 bash tools/nn/dock.sh ...   names the container (so it can be told apart and stopped).
# DOCK_CPUS=8 bash tools/nn/dock.sh ...   caps the container at 8 CPU cores (while the user plays; a running
#   container: docker update --cpus 8 <name>) and torch's threads to as many (OMP_NUM_THREADS, MKL_NUM_THREADS:
#   torch otherwise starts a thread per host core, which a capped container only time-slices).
# torch.compile keeps its kernels between runs in the Docker volume tww3-torch-cache: a run that
# compiles what an earlier one did loads it (the warm-up of training and evaluation gets shorter).
# DOCK_BUILD=<folder> bash tools/nn/dock.sh ...   mounts that folder as the repo's build/ (a worktree's run that
#   writes into the main checkout's build: tools/ops/baselines.py).
repo="$(cd "$(dirname "$0")/../.." && pwd -W 2>/dev/null || pwd)"
build=()
if [ -n "$DOCK_BUILD" ]; then build=(-v "$(cd "$DOCK_BUILD" && (pwd -W 2>/dev/null || pwd)):/repo/build"); fi
threads=()
if [ -n "$DOCK_CPUS" ]; then
  n="${DOCK_CPUS%%.*}"
  [ "${n:-0}" -ge 1 ] 2>/dev/null || n=1
  threads=(-e "OMP_NUM_THREADS=$n" -e "MKL_NUM_THREADS=$n")
fi
MSYS_NO_PATHCONV=1 exec docker run --rm ${DOCK_NAME:+--name "$DOCK_NAME"} ${DOCK_CPUS:+--cpus "$DOCK_CPUS"} --gpus all --shm-size 2g -v "${repo}:/repo" "${build[@]}" "${threads[@]}" -w /repo -e PYTHONPATH=/repo \
  -v tww3-torch-cache:/cache -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor -e TRITON_CACHE_DIR=/cache/triton \
  -e PYTHONUNBUFFERED=1 snake-ai-trainer python -m "$@"
