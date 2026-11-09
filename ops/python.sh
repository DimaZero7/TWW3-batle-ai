#!/usr/bin/env bash
# The host's python for the scripts (sourced): $PY - the project's .venv, else the old checkout's beside it
# (../TWW3-batle-ai/.venv, while the new tree is assembled next to it), else python on PATH.
if [ -z "$PY" ]; then
  for p in .venv/Scripts/python.exe .venv/bin/python ../TWW3-batle-ai/.venv/Scripts/python.exe; do
    if [ -x "$p" ]; then PY="$p"; break; fi
  done
  PY="${PY:-python}"
fi
