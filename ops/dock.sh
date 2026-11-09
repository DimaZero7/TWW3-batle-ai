#!/bin/bash
# Runs a Python module of this project in the training container (PyTorch + CUDA), the project's root on /repo:
#   bash ops/dock.sh pytest -q sim/tests            (any module: python -m "$@")
# The image is snake-ai-trainer (PyTorch 2.13, CUDA 13.2, Python 3.12; ops/README.md).
# DOCK_NAME=agent-<task> bash ops/dock.sh ...   names the container (so it can be told apart and stopped).
# DOCK_CPUS=6 bash ops/dock.sh ...   caps the container at 6 CPU cores (while a battle or a training runs; a running
#   container: docker update --cpus 6 <name>) and torch's threads to as many (OMP_NUM_THREADS, MKL_NUM_THREADS:
#   torch otherwise starts a thread per host core, which a capped container only time-slices).
# DOCK_GPU=0 bash ops/dock.sh ...   no GPU (--gpus all by default).
# torch.compile keeps its kernels between runs in the Docker volume tww3-torch-cache: a run that
# compiles what an earlier one did loads it (the warm-up of training and evaluation gets shorter).
# DOCK_MEM=8g bash ops/dock.sh ...   caps the container's memory (docker run --memory; the machine's 32 GB are shared with the user).
# DOCK_BUILD=<folder> bash ops/dock.sh ...   mounts that folder as the project's build/ (a copy's run that writes
#   into the main checkout's build).
repo="$(cd "$(dirname "$0")/.." && pwd -W 2>/dev/null || pwd)"
build=()
if [ -n "$DOCK_BUILD" ]; then build=(-v "$(cd "$DOCK_BUILD" && (pwd -W 2>/dev/null || pwd)):/repo/build"); fi
threads=()
if [ -n "$DOCK_CPUS" ]; then
  n="${DOCK_CPUS%%.*}"
  [ "${n:-0}" -ge 1 ] 2>/dev/null || n=1
  threads=(-e "OMP_NUM_THREADS=$n" -e "MKL_NUM_THREADS=$n")
fi
gpus=(--gpus all)
if [ "${DOCK_GPU:-1}" = "0" ]; then gpus=(); fi
MSYS_NO_PATHCONV=1 exec docker run --rm ${DOCK_NAME:+--name "$DOCK_NAME"} ${DOCK_CPUS:+--cpus "$DOCK_CPUS"} ${DOCK_MEM:+--memory "$DOCK_MEM"} "${gpus[@]}" --shm-size 2g -v "${repo}:/repo" "${build[@]}" "${threads[@]}" -w /repo -e PYTHONPATH=/repo \
  -v tww3-torch-cache:/cache -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor -e TRITON_CACHE_DIR=/cache/triton \
  -e PYTHONUNBUFFERED=1 snake-ai-trainer python -m "$@"
