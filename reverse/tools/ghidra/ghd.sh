#!/bin/bash
# usage: ghd.sh <AnalyzeHeadless args...>   Ghidra 12.1.4 and JDK 21 live in reverse/ghidra/ (outside git)
R="$(cd "$(dirname "$0")/../.." && pwd -W)"
G="$R/ghidra/ghidra_12.1.4_PUBLIC"
exec "$R/ghidra/jdk21/bin/java.exe" -Xmx12G -Djava.system.class.loader=ghidra.GhidraClassLoader -Dfile.encoding=UTF8 -Duser.country=US -Duser.language=en -Djava.awt.headless=true -Xshare:off --enable-native-access=ALL-UNNAMED -Dlog4j.skipJansi=true -cp "$G/Ghidra/Framework/Utility/lib/Utility.jar" ghidra.Ghidra ghidra.app.util.headless.AnalyzeHeadless "$@"
