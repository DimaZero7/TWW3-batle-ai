"""The page's scripted check: headless Chrome driven through its DevTools protocol (host python, stdlib only - a tiny
websocket client here; no playwright on this machine). The server must run (bash tools/label/serve.sh).

    python -m tools.label.browser_check [--url http://localhost:8790] [--out build/diag_label] [--chrome PATH]

The scenario: build a matchup (3 + 3 from the pool, placed by dragging one unit), save it, start as the operative,
on the first pause take the formula's hint, then at the next pause re-cut the groups with the mouse (a box over two
units -> group 2) and drag group 2's zone, decide, run to the end at "until the decision" with 'keep', and read the
record's folder back; then the tactician on the captured map ksl_plains_01 (the enemy at a run): the formula's frames
at the first pause, one frame dragged and shrunk at the second, 'keep' to the end. Screenshots: setup.png,
decision.png, end.png, decision_tactic.png, decision_strategist.png in --out (the strategist
at the end: the formula's deployment with group 1's zone moved 120 m left); a json of what was checked on stdout.
"""
import argparse
import base64
import json
import os
import socket
import struct
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

CHROMES = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
           r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"]


class WS:
    """A minimal websocket client (RFC 6455, text frames, client masking) for the DevTools protocol."""

    def __init__(self, url):
        assert url.startswith("ws://")
        hostport, path = url[5:].split("/", 1)
        host, port = hostport.split(":")
        self.s = socket.create_connection((host, int(port)), timeout=60)
        key = base64.b64encode(os.urandom(16)).decode()
        req = (f"GET /{path} HTTP/1.1\r\nHost: {hostport}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
               f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n")
        self.s.sendall(req.encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            buf += self.s.recv(4096)
        assert b" 101 " in buf.split(b"\r\n")[0], buf[:200]
        self.rest = buf.split(b"\r\n\r\n", 1)[1]
        self.n = 0

    def _recv(self, k):
        while len(self.rest) < k:
            self.rest += self.s.recv(1 << 20)
        out, self.rest = self.rest[:k], self.rest[k:]
        return out

    def send(self, obj):
        data = json.dumps(obj).encode()
        head = bytearray([0x81])
        n = len(data)
        if n < 126:
            head.append(0x80 | n)
        elif n < 65536:
            head += bytes([0x80 | 126]) + struct.pack(">H", n)
        else:
            head += bytes([0x80 | 127]) + struct.pack(">Q", n)
        mask = os.urandom(4)
        self.s.sendall(bytes(head) + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))

    def recv(self):
        msg = b""
        while True:
            b0, b1 = self._recv(2)
            n = b1 & 0x7F
            if n == 126:
                n = struct.unpack(">H", self._recv(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self._recv(8))[0]
            msg += self._recv(n)
            if b0 & 0x80:
                return json.loads(msg.decode())

    def call(self, method, **params):
        self.n += 1
        self.send({"id": self.n, "method": method, "params": params})
        while True:
            m = self.recv()
            if m.get("id") == self.n:
                if "error" in m:
                    raise RuntimeError(f"{method}: {m['error']}")
                return m.get("result", {})


class Page:
    def __init__(self, ws):
        self.ws = ws

    def js(self, expr, wait=True):
        r = self.ws.call("Runtime.evaluate", expression=expr, awaitPromise=wait, returnByValue=True)
        if "exceptionDetails" in r:
            raise RuntimeError(f"js: {expr[:80]}: {r['exceptionDetails'].get('exception', {}).get('description')}")
        return r.get("result", {}).get("value")

    def until(self, expr, timeout=180.0, step=0.25):
        t0 = time.time()
        while time.time() - t0 < timeout:
            v = self.js(expr)
            if v:
                return v
            time.sleep(step)
        raise TimeoutError(expr)

    def mouse(self, kind, x, y, button="left", buttons=1):
        self.ws.call("Input.dispatchMouseEvent", type=kind, x=x, y=y, button=button, buttons=buttons, clickCount=1)

    def drag(self, a, b, steps=8, button="left"):
        self.mouse("mouseMoved", *a, button="none", buttons=0)
        self.mouse("mousePressed", *a, button=button)
        for k in range(1, steps + 1):
            x = a[0] + (b[0] - a[0]) * k / steps
            y = a[1] + (b[1] - a[1]) * k / steps
            self.mouse("mouseMoved", x, y, button=button)
        self.mouse("mouseReleased", *b, button=button)

    def click(self, x, y, button="left"):
        self.mouse("mousePressed", x, y, button=button)
        self.mouse("mouseReleased", x, y, button=button)

    def shot(self, path):
        r = self.ws.call("Page.captureScreenshot", format="png")
        Path(path).write_bytes(base64.b64decode(r["data"]))


def screen_of(page, x, z):
    """The page's canvas point (CSS px of the viewport) of world (x, z)."""
    return page.js(f"(()=>{{const r=cv.getBoundingClientRect();const [sx,sy]=toS({x},{z});return [r.left+sx,r.top+sy];}})()")


def run(url, out, chrome):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    prof = tempfile.mkdtemp(prefix="label_chrome_")
    port = 9333
    proc = subprocess.Popen([chrome, "--headless=new", f"--remote-debugging-port={port}", f"--user-data-dir={prof}",
                             "--window-size=1500,900", "--force-device-scale-factor=1", "--no-first-run",
                             "--no-default-browser-check", "about:blank"], stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    checked = {}
    try:
        for _ in range(60):
            try:
                tabs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/json").read())
                break
            except OSError:
                time.sleep(0.5)
        tab = next(t for t in tabs if t.get("type") == "page")
        page = Page(WS(tab["webSocketDebuggerUrl"]))
        page.ws.call("Page.enable")
        page.ws.call("Emulation.setDeviceMetricsOverride", width=1500, height=900, deviceScaleFactor=1, mobile=False)
        page.ws.call("Page.navigate", url=url)
        page.until("typeof S!=='undefined' && S.pool!==null && S.m.sides[1].units.length===3")
        # the matchup: one more unit from the pool search, one unit dragged, saved
        page.js("document.getElementById('pool-q').value='halberd'; renderPool(); "
                "document.querySelector('#pool .item').click(); true")
        u = page.js("S.m.sides[1].units[0]")
        a = screen_of(page, u["x"], u["z"])
        page.drag(a, (a[0] - 40, a[1] - 20))
        moved = page.js("S.m.sides[1].units[0]")
        checked["dragged_m"] = round(((moved["x"] - u["x"]) ** 2 + (moved["z"] - u["z"]) ** 2) ** 0.5, 1)
        page.js("document.getElementById('m-name').value='проверка'; document.getElementById('m-limit').value=150; "
                "document.getElementById('btn-save').click(); true")
        page.until("document.getElementById('m-list').value==='проверка'", 20)
        checked["saved"] = page.js("document.getElementById('m-list').value")
        page.shot(out / "setup.png")
        # start as the operative; the first pause: the formula's hint
        page.js("document.getElementById('s-level').value='operative'; levelChanged(); "
                "document.getElementById('btn-start').click(); true")
        page.until("S.pending && S.pending.sec===0 && S.pv", 120)
        checked["first_pause_s"] = page.js("S.pending.sec")
        page.js("document.getElementById('btn-hint').click(); true")
        page.until("S.pv && S.pv.ok", 30)
        page.js("document.getElementById('btn-next').click(); true")
        page.until("S.pending && S.pending.sec===15 && S.pv", 180)
        checked["second_pause_s"] = page.js("S.pending.sec")
        # re-cut: a box over my two right-most units -> group 2 (the key '2'); drag group 2's goal forward
        mine = page.js("S.shown.units.filter(u=>u[1]===1&&u[10]).map(u=>[u[0],u[2],u[3]]).sort((a,b)=>b[1]-a[1])")
        two = mine[:2]
        xs = [p[1] for p in two]
        zs = [p[2] for p in two]
        p0 = screen_of(page, min(xs) - 25, max(zs) + 25)
        p1 = screen_of(page, max(xs) + 25, min(zs) - 25)
        page.drag(p0, p1)
        checked["box_selected"] = page.js("[...S.sel]")
        page.ws.call("Input.dispatchKeyEvent", type="keyDown", key="2", text="2")
        page.ws.call("Input.dispatchKeyEvent", type="keyUp", key="2")
        page.until("S.pv && S.pv.groups && S.pv.groups.some(g=>g.g===1)", 30)
        g1 = page.js("S.pv.groups.find(g=>g.g===1)")
        gl = g1["goal"] or [g1["cx"], g1["cz"]]
        a = screen_of(page, gl[0], gl[1])
        page.js("S.activeG=1; true")
        page.click(a[0] + 0, a[1] - 60, button="right")          # the right button: the goal 60 px forward
        page.until("S.pv && S.pv.groups && S.pv.groups.find(g=>g.g===1) && S.pv.groups.find(g=>g.g===1).goal", 30)
        time.sleep(0.6)
        checked["preview_ok"] = page.js("S.pv.ok")
        checked["preview_errors"] = page.js("S.pv.errors")
        checked["groups_after"] = page.js("S.pv.groups.map(g=>({g:g.g,members:g.members,cx:g.cx,cz:g.cz,task:g.task}))")
        page.shot(out / "decision.png")
        if not checked["preview_ok"]:
            # the preview names the ban (e.g. zone_sep_m): take the zone back where the group stands
            page.js("for(const e of S.edit.groups){if(e.task) e.task.goal=null;} preview(); true")
            time.sleep(1.5)
            checked["preview_ok_retry"] = page.js("S.pv.ok")
        page.js("document.getElementById('btn-next').click(); true")
        page.until("!S.pending || S.pending.sec>15", 60)
        checked["after_decide_status"] = page.js("document.getElementById('status').textContent")
        # to the end: 'keep' at every pause
        page.js("S.lastMode='decision'; setMode('decision'); true")
        t0 = time.time()
        while time.time() - t0 < 900:
            if page.js("!!document.querySelector('#result:not(.hidden)')"):
                break
            if page.js("!!S.pending && !!S.pv"):
                page.js("document.getElementById('btn-keep').click(); true")
            time.sleep(0.4)
        checked["result"] = page.js("S.st.result")
        checked["folder"] = page.js("S.st.folder")
        page.js("const r=document.getElementById('replay-t'); r.value=Math.floor(r.max/2); r.oninput({target:r}); true")
        time.sleep(0.3)
        page.shot(out / "end.png")
        # the tactician on a captured map: the same matchup on ksl_plains_01, enemy at a run, 60 s
        page.js("showTab('setup'); document.getElementById('m-map').value='ksl_plains_01'; setMap('ksl_plains_01'); "
                "document.getElementById('m-enemy').value='attack_run'; document.getElementById('m-limit').value=60; "
                "document.getElementById('s-level').value='tactic'; levelChanged(); "
                "document.getElementById('btn-start').click(); true")
        page.until("S.st && S.st.level==='tactic' && S.pending && S.pending.sec===0 && S.pv", 180)
        checked["tactic_first_pause"] = page.js("({sec:S.pending.sec, rows:S.pending.rows.length, can_keep:S.pending.can_keep})")
        page.js("document.getElementById('btn-next').click(); true")
        page.until("S.pending && S.pending.level==='tactic' && S.pending.sec===5 && S.pv", 120)
        # drag the first unit's frame 30 m to the right and shrink it to 48 x 48
        r0 = page.js("S.pending.rows[0]")
        e0 = page.js("S.edit.rows[0].units[0]")
        page.js("S.edit.rows[0].units[0].w=48; S.edit.rows[0].units[0].d=48; S.edit.rows[0].units[0].keep=false; true")
        a = screen_of(page, e0["x"], e0["z"])
        page.drag(a, (a[0] + 30 * page.js("S.cam.scale"), a[1]))
        time.sleep(1.0)
        page.until("S.pv && S.pv.frames", 30)
        checked["tactic_edit"] = page.js("S.edit.rows[0].units[0]")
        checked["tactic_preview"] = page.js("({ok:S.pv.ok, errors:S.pv.errors, warnings:S.pv.warnings, "
                                            "frame:S.pv.frames[String(S.pending.rows[0].g)]})")
        checked["tactic_zone"] = {k: r0[k] for k in ("g", "cx", "cz", "heading", "front", "depth")}
        if not checked["tactic_preview"]["ok"]:
            # refused (the frame laps over its neighbour): drag it back past its place, 15 m to the left of the start
            e1 = page.js("S.edit.rows[0].units[0]")
            a = screen_of(page, e1["x"], e1["z"])
            page.drag(a, (a[0] - 45 * page.js("S.cam.scale"), a[1]))
            time.sleep(1.0)
            checked["tactic_preview_retry"] = page.js("({ok:S.pv.ok, errors:S.pv.errors, "
                                                      "frame:S.pv.frames[String(S.pending.rows[0].g)]})")
        page.shot(out / "decision_tactic.png")
        page.js("document.getElementById('btn-next').click(); true")
        page.until("!S.pending || S.pending.sec>5", 60)
        t0 = time.time()
        while time.time() - t0 < 900:
            if page.js("!!document.querySelector('#result:not(.hidden)')"):
                break
            if page.js("!!S.pending && !!S.pv"):
                page.js("document.getElementById('btn-keep').click(); true")
            time.sleep(0.4)
        checked["tactic_result"] = page.js("S.st.result")
        checked["tactic_folder"] = page.js("S.st.folder")
        # the strategist on the flat field: the formula's deployment, group 1's zone moved by the right button
        page.js("showTab('setup'); document.getElementById('m-map').value='flat'; setMap('flat'); "
                "document.getElementById('m-enemy').value='attack_run'; document.getElementById('m-limit').value=60; "
                "document.getElementById('s-level').value='strategist'; levelChanged(); "
                "document.getElementById('btn-start').click(); true")
        page.until("S.st && S.st.level==='strategist' && S.pending && S.pending.level==='strategist' && S.pv", 180)
        g0 = page.js("S.pv.groups[0]")
        page.js(f"S.activeG={g0['g']}; true")
        a = screen_of(page, g0["cx"] - 120.0, g0["cz"])
        page.click(a[0], a[1], button="right")
        time.sleep(1.5)
        page.until("S.pv && S.pv.places", 30)
        checked["strategist_preview"] = page.js("({ok:S.pv.ok, errors:S.pv.errors, groups:S.pv.groups.map(g=>"
                                                "({g:g.g,members:g.members,cx:g.cx,cz:g.cz}))})")
        page.shot(out / "decision_strategist.png")
        page.js("document.getElementById('btn-next').click(); true")
        page.until("!!document.querySelector('#result:not(.hidden)')", 600)
        checked["strategist_result"] = page.js("S.st.result")
        checked["strategist_folder"] = page.js("S.st.folder")
    except Exception as e:                       # what was checked so far, the error and the page's own messages
        checked["error"] = f"{type(e).__name__}: {e}"
        try:
            checked["page_errors"] = page.js("document.getElementById('d-err').textContent + ' | ' + "
                                             "document.getElementById('status').textContent")
            page.shot(out / "failed.png")
        except Exception:
            pass
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
    return checked


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://localhost:8790")
    ap.add_argument("--out", default="build/diag_label")
    ap.add_argument("--chrome", default=next((c for c in CHROMES if Path(c).exists()), None))
    a = ap.parse_args(argv)
    print(json.dumps(run(a.url, a.out, a.chrome), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
