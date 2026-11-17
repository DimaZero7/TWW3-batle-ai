"""Hooks around context compaction; wired in .claude/settings.json.

  compact.py pre           PreCompact: prints compact_keep.md - Claude Code takes stdout as instructions for the summary
  compact.py start <file>  SessionStart (compact, clear): prints one project memory file - stdout joins the new context
  compact.py post          PostCompact: saves the produced summary to tmp/handoff/compact_<time>_<session>_<trigger>.md

Every call appends one line to tmp/handoff/hooks.log. A hook must never break a session, so every path ends with
exit code 0 (code 2 from PreCompact would block the compaction). Stdout over 10,000 characters is cut by Claude Code,
so the output is held under CAP.
"""
import json
import re
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = ROOT / "tmp" / "handoff"
CAP = 9500


def note(event, data, text):
    """One line per call: when, which hook, which session, what it did."""
    try:
        OUT.mkdir(parents=True, exist_ok=True)
        kind = data.get("trigger") or data.get("source") or "-"
        line = "%s %s %s session=%s %s\n" % (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), event, kind, str(data.get("session_id", "-"))[:8], text)
        with open(OUT / "hooks.log", "a", encoding="utf-8") as f:
            f.write(line)
    except OSError:
        pass


def memory_dir(data):
    """The project's memory folder: beside the session's transcript, else by the project path."""
    path = data.get("transcript_path")
    if path:
        beside = Path(path).parent / "memory"
        if beside.is_dir():
            return beside
    return Path.home() / ".claude" / "projects" / re.sub(r"[^A-Za-z0-9]", "-", str(ROOT)) / "memory"


def pre_ran(data, minutes=30):
    """Did the PreCompact hook of this session run just now? Read back from hooks.log, for the line after a compaction."""
    mark = " PreCompact "
    tag = "session=%s " % str(data.get("session_id", "-"))[:8]
    try:
        lines = (OUT / "hooks.log").read_text(encoding="utf-8").splitlines()[-200:]
    except OSError:
        lines = []
    for line in reversed(lines):
        if mark in line and tag in line:
            try:
                age = (datetime.now() - datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")).total_seconds()
            except ValueError:
                continue
            if 0 <= age <= minutes * 60:
                return "the PreCompact hook ran at %s, so the summary was written with the project's keep-list" % line[11:19]
            break
    return ("NO PreCompact record for this session in the last %d minutes: the summary was probably written without "
            "the project's keep-list (.claude/hooks/compact_keep.md) - tell the user" % minutes)


def pre(data):
    text = (HERE / "compact_keep.md").read_text(encoding="utf-8").strip()[:CAP]
    sys.stdout.write(text + "\n")
    note("PreCompact", data, "instructions chars=%d" % len(text))


def start(data, name):
    path = memory_dir(data) / name
    body = path.read_text(encoding="utf-8")
    body = re.sub(r"\A---\n.*?\n---\n", "", body, flags=re.S).strip()
    when = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    head = ("[Project memory re-injected after '%s': %s, last edited %s. It is the project's recorded state and the "
            "orchestrator's working rules, not a new instruction; your role stays what it was. Check it against "
            "git and the clock before acting.]\n" % (data.get("source", "?"), path, when))
    if data.get("source") == "compact":
        head += "[Hooks at this compaction: %s. Saved summaries and the hook log: %s]\n" % (pre_ran(data), OUT)
    head += "\n"
    if len(head) + len(body) > CAP:
        body = body[:CAP - len(head) - 80] + "\n[cut here - read the whole file: %s]" % path
    sys.stdout.write(head + body + "\n")
    note("SessionStart", data, "%s chars=%d" % (name, len(head) + len(body)))


def post(data):
    summary = data.get("compact_summary") or ""
    if not summary.strip():
        note("PostCompact", data, "no summary in the input - nothing saved")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    name = "compact_%s_%s_%s.md" % (
        datetime.now().strftime("%Y%m%d-%H%M%S"), str(data.get("session_id", "nosession"))[:8],
        data.get("trigger", "unknown"))
    head = "<!-- session %s | trigger %s | transcript %s -->\n\n" % (
        data.get("session_id"), data.get("trigger"), data.get("transcript_path"))
    (OUT / name).write_text(head + summary + "\n", encoding="utf-8")
    note("PostCompact", data, "%s chars=%d" % (name, len(summary)))


def main():
    data = {}
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        sys.stdout.reconfigure(encoding="utf-8", newline="\n")
        raw = sys.stdin.buffer.read().decode("utf-8", "replace").strip()
        try:
            data = json.loads(raw) if raw else {}
        except ValueError as err:  # the instructions and the memory files do not depend on the input
            note("WARN", {}, "%s input is not JSON (%s) - going on without it" % (mode, err))
        if not isinstance(data, dict):
            data = {}
        if mode == "pre":
            pre(data)
        elif mode == "start" and len(sys.argv) > 2:
            start(data, sys.argv[2])
        elif mode == "post":
            post(data)
        else:
            note("unknown", data, "argv=%r" % (sys.argv[1:],))
    except Exception as err:  # a broken hook must stay silent and harmless
        note("ERROR", data if isinstance(data, dict) else {}, "%s %s: %s" % (mode, type(err).__name__, err))
    sys.exit(0)


if __name__ == "__main__":
    main()
