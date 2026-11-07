"""tools/label/keys.ps1: the game's process never goes into $game - PowerShell's names ignore case and $Game is the
[string] folder parameter (a process put there became its name, the window handle $null, and the helper died as the game
started: Ctrl never reached the bridge, the person's battle 07.11 20:23)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_the_helper_keeps_the_process_apart_from_the_folder_parameter():
    text = (ROOT / "tools" / "label" / "keys.ps1").read_text(encoding="utf-8")
    assert re.search(r"\[string\]\$Game\b", text)
    assert not re.search(r"(?i)^\s*\$game\s*=", text, re.M)
    assert "try { Write-Keys } catch" in text
