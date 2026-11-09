#!/bin/bash
# usage: dec.sh out.c addr...   decompile functions at the addresses into out.c (the project is opened read-only)
# GPROJ = the project folder (default reverse/ghidra/gproj_exe1015, exe 15.10); a second parallel run needs its own copy
T="$(cd "$(dirname "$0")" && pwd -W)"
R="$(cd "$T/../.." && pwd -W)"
GPROJ="${GPROJ:-$R/ghidra/gproj_exe1015}"
out="$1"; shift
rm -f "$out" 2>/dev/null
"$T/ghd.sh" "$GPROJ" tww3 -process Warhammer3.exe -noanalysis -readOnly -scriptPath "$T" -postScript Decomp.java "$out" "$@" > "$R/ghidra/dec.log" 2>&1
grep -c "=====" "$out"
