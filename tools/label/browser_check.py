"""The page's scripted check: headless Chrome driven through its DevTools protocol (host python, stdlib only - a tiny
websocket client here; no playwright on this machine). The server must run (bash tools/label/serve.sh).

    python -m tools.label.browser_check [--url http://localhost:8790] [--out build/diag_label] [--chrome PATH]

The scenario, through the page's own buttons and the mouse: the setup's five steps (the first example, halberdiers
added by the search and '+', the map, a unit dragged and widened by its edge, the role, the summary), the operative
(the formula's hint taken at 0 s; at 15 s 'Поменять', a box over two units -> group 2 by the key, its zone dragged, its
arrow turned; then 'Оставить как есть' to the end), the tactician on ksl_river_01 (a frame dragged and shrunk), the
strategist (a group's zone dragged). Screenshots v2_*.png in --out; a json of what was checked on stdout.
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


def click_text(page, text, scope="document"):
    """Clicks the first visible button whose text starts with text (a real click at its middle)."""
    r = page.js(f"""(()=>{{const b=[...{scope}.querySelectorAll('button')].find(b=>b.offsetParent&&!b.disabled&&
        b.textContent.trim().startsWith({json.dumps(text)}));if(!b)return null;b.scrollIntoView({{block:'center'}});
        const r=b.getBoundingClientRect();
        return [r.left+r.width/2,r.top+r.height/2];}})()""")
    if r is None:
        raise RuntimeError(f"no button {text!r}")
    page.click(*r)
    time.sleep(0.25)


def unit_screen(page, slot):
    return page.js(f"(()=>{{const o=S.drawn.find(o=>o.slot==={slot});const r=cv.getBoundingClientRect();"
                   f"const [x,y]=toS(o.x,o.z);return [r.left+x,r.top+y];}})()")


def play_to_end(page, checked, key, limit=900):
    t0 = time.time()
    n = 0
    while time.time() - t0 < limit:
        if page.js("!!document.querySelector('#result:not(.hidden)')"):
            break
        if page.js("!!S.pending && !!S.pv && !S.changing"):
            if page.js("S.pending.can_keep"):
                click_text(page, "Оставить как есть", "document.getElementById('banner')")
            else:
                click_text(page, "Принять подсказку", "document.getElementById('banner')")
            n += 1
        time.sleep(0.4)
    checked[key] = page.js("S.st.result")
    checked[key + "_clicks"] = n


def run(url, out, chrome):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    prof = tempfile.mkdtemp(prefix="label_chrome_")
    port = 9333
    proc = subprocess.Popen([chrome, "--headless=new", f"--remote-debugging-port={port}", f"--user-data-dir={prof}",
                             "--window-size=1600,950", "--force-device-scale-factor=1", "--no-first-run",
                             "--no-default-browser-check", "about:blank"], stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    checked = {}
    page = None
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
        page.ws.call("Emulation.setDeviceMetricsOverride", width=1600, height=950, deviceScaleFactor=1, mobile=False)
        page.ws.call("Page.navigate", url=url)
        page.until("typeof S!=='undefined' && S.pool && S.m && S.m.sides[1].units.length===3", 60)
        time.sleep(1.0)
        # 1. the armies: the first example is in; add halberdiers to mine by the search and '+'
        page.js("(()=>{const i=document.querySelector('.col.mine input');i.value='алебард';i.dispatchEvent(new Event('input'));})()")
        time.sleep(0.3)
        page.js("document.querySelectorAll('.col.mine .list')[1].querySelector('button').click()")
        time.sleep(0.4)
        checked["armies"] = page.js("[S.m.sides[1].units.map(u=>u.key), S.m.sides[2].units.length, gold(1), gold(2)]")
        page.shot(out / "v2_1_armies.png")
        # 2. the map
        click_text(page, "Дальше")
        time.sleep(0.6)
        page.shot(out / "v2_2_map.png")
        # 3. the deployment: drag the first unit, then widen it by its edge handle
        click_text(page, "Дальше")
        time.sleep(0.6)
        u = page.js("S.m.sides[1].units[1]")
        a = screen_of(page, u["x"], u["z"])
        page.drag(a, (a[0] + 50, a[1] - 15))
        u2 = page.js("S.m.sides[1].units[1]")
        checked["dragged_m"] = round(((u2["x"] - u["x"]) ** 2 + (u2["z"] - u["z"]) ** 2) ** 0.5, 1)
        o = page.js("(()=>{const o=setupUnits().find(o=>o.side===1&&o.i===1);return {h:widthHandles(o),front:o.front};})()")
        h = screen_of(page, *o["h"][0])
        page.drag(h, (h[0] + 30 * page.js("S.cam.scale"), h[1]))
        checked["width_before_after"] = [round(o["front"], 1),
                                         page.js("Math.round(setupUnits().find(o=>o.side===1&&o.i===1).front)")]
        page.shot(out / "v2_3_deploy.png")
        # 4. the role, 5. the summary
        click_text(page, "Дальше")
        page.js("S.level='operative'; S.period=15; S.m.limit_s=90; renderSetup(); true")
        page.shot(out / "v2_4_role.png")
        click_text(page, "Дальше")
        time.sleep(0.4)
        page.shot(out / "v2_5_go.png")
        click_text(page, "В бой")
        # the operative: the first pause - take the formula's hint
        page.until("S.pending && S.pending.sec===0 && S.pv && S.hintPv", 240)
        page.shot(out / "v2_6_op_first.png")
        click_text(page, "Принять подсказку", "document.getElementById('banner')")
        page.until("S.pending && S.pending.sec===15 && S.pv", 300)
        page.shot(out / "v2_7_op_pause.png")
        click_text(page, "Поменять", "document.getElementById('banner')")
        # a box over my two right-most units -> group 2 (key '2'), then drag group 2's zone and its arrow
        mine = page.js("S.drawn.filter(o=>o.side===1).map(o=>[o.slot,o.x,o.z]).sort((a,b)=>b[1]-a[1])")
        two = mine[:2]
        xs, zs = [p[1] for p in two], [p[2] for p in two]
        # (from below the units: their cards stand above them and a press on a card selects that unit)
        p0 = screen_of(page, min(xs) - 30, min(zs) - 40)
        p1 = screen_of(page, max(xs) + 30, max(zs) + 12)
        page.drag(p0, p1)
        checked["box_selected"] = page.js("[...S.sel]")
        page.ws.call("Input.dispatchKeyEvent", type="keyDown", key="2", text="2")
        page.ws.call("Input.dispatchKeyEvent", type="keyUp", key="2")
        page.until("S.pv && S.pv.groups && S.pv.groups.some(g=>g.g===1)", 30)
        time.sleep(0.5)
        g1 = page.js("S.pv.groups.find(g=>g.g===1)")
        a = screen_of(page, g1["cx"], g1["cz"])
        page.drag(a, (a[0] + 60, a[1]))
        time.sleep(1.0)
        g1 = page.js("S.pv.groups.find(g=>g.g===1)")
        tip = page.js(f"zoneArrow(S.pv.groups.find(g=>g.g===1))")
        t = screen_of(page, *tip)
        page.drag(t, (t[0] + 18, t[1] + 4))
        time.sleep(1.2)
        checked["op_edit"] = page.js("({ok:S.pv.ok, errors:S.pv.errors, groups:S.pv.groups.map(g=>({g:g.g,members:g.members,"
                                     "cx:g.cx,cz:g.cz,heading:g.heading})), did:S.did})")
        page.shot(out / "v2_8_op_edit.png")
        if not checked["op_edit"]["ok"]:
            page.shot(out / "v2_8b_op_refused.png")
            checked["op_refused_text"] = page.js("document.getElementById('d-err').innerText")
            page.js("for(const e of S.edit.groups){if(e.task){e.task.goal=null; e.task.turn=0;}} preview(); true")
            time.sleep(1.5)
            checked["op_edit_retry_ok"] = page.js("S.pv.ok")
        click_text(page, "Дальше ▶")
        page.until("!S.pending || S.pending.sec>15", 60)
        checked["after_decide"] = page.js("document.getElementById('status').textContent")
        page.js("S.lastMode='decision'; setMode('decision'); true")
        play_to_end(page, checked, "op_result")
        checked["op_folder"] = page.js("S.st.folder")
        page.shot(out / "v2_9_result.png")
        # the tactician on the river map
        page.js("toSetup(); setMap('ksl_river_01'); S.m.enemy='attack_run'; S.level='tactic'; S.period=5; "
                "S.m.limit_s=40; true")
        page.js("startBattle(); true")
        page.until("S.st && S.st.level==='tactic' && S.pending && S.pending.sec===0 && S.pv", 240)
        click_text(page, "Принять подсказку", "document.getElementById('banner')")
        page.until("S.pending && S.pending.level==='tactic' && S.pending.sec===5 && S.pv", 240)
        click_text(page, "Поменять", "document.getElementById('banner')")
        e0 = page.js("S.edit.rows[0].units[0]")
        page.js("S.edit.rows[0].units[0].w=48; S.edit.rows[0].units[0].d=48; true")
        a = screen_of(page, e0["x"], e0["z"])
        page.drag(a, (a[0] - 15 * page.js("S.cam.scale"), a[1]))
        time.sleep(1.2)
        checked["tactic_preview"] = page.js("({ok:S.pv.ok, errors:S.pv.errors})")
        page.shot(out / "v2_10_tactic.png")
        if not checked["tactic_preview"]["ok"]:
            page.js("S.edit=JSON.parse(JSON.stringify(S.pending.edit)); preview(); true")
            time.sleep(1.2)
        click_text(page, "Дальше ▶")
        play_to_end(page, checked, "tactic_result")
        # the strategist on the flat field
        page.js("toSetup(); setMap('flat'); S.m.enemy='attack'; S.level='strategist'; S.m.limit_s=40; true")
        page.js("startBattle(); true")
        page.until("S.st && S.st.level==='strategist' && S.pending && S.pv", 240)
        click_text(page, "Поменять", "document.getElementById('banner')")
        g0 = page.js("S.pv.groups[0]")
        a = screen_of(page, g0["cx"], g0["cz"])
        page.drag(a, (a[0] - 80, a[1]))
        time.sleep(1.5)
        checked["strategist_preview"] = page.js("({ok:S.pv.ok, errors:S.pv.errors, groups:S.pv.groups.map(g=>({g:g.g,cx:g.cx,cz:g.cz}))})")
        page.shot(out / "v2_11_strategist.png")
        click_text(page, "Дальше ▶")
        play_to_end(page, checked, "strategist_result")
    except Exception as e:                       # what was checked so far, the error and the page's own messages
        checked["error"] = f"{type(e).__name__}: {e}"
        try:
            checked["page_errors"] = page.js("document.getElementById('d-err').innerText + ' | ' + "
                                             "document.getElementById('status').textContent")
            page.shot(out / "v2_failed.png")
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
