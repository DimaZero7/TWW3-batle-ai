#!/bin/bash
# Разметка: the labelling server in the training container (torch; the simulator on the CPU, no GPU), the page at
# http://localhost:8790 (the port published on 127.0.0.1 only). Ctrl+C or `docker stop label` ends it.
#   bash tools/label/serve.sh
#   LABEL_PORT=8766 DOCK_NAME=label2 DOCK_CPUS=4 bash tools/label/serve.sh
repo="$(cd "$(dirname "$0")/../.." && pwd -W 2>/dev/null || pwd)"
port="${LABEL_PORT:-8790}"
name="${DOCK_NAME:-label}"
cpus="${DOCK_CPUS:-4}"
n="${cpus%%.*}"
tty=()
if [ -t 0 ] && [ -t 1 ]; then tty=(-it); fi
echo "Разметка: http://localhost:${port}  (контейнер ${name}; стоп - Ctrl+C или docker stop ${name})"
MSYS_NO_PATHCONV=1 exec docker run --rm --init "${tty[@]}" --name "$name" --cpus "$cpus" \
  -e "OMP_NUM_THREADS=$n" -e "MKL_NUM_THREADS=$n" -p "127.0.0.1:${port}:8790" \
  -v "${repo}:/repo" -w /repo -e PYTHONPATH=/repo -e PYTHONUNBUFFERED=1 \
  snake-ai-trainer python -m tools.label.server --port 8790 "$@"
