"""The scripted check of Разметка в игре: the page in headless Chrome (browser_check's DevTools client) plays the person
while tools/label/game.ps1 runs the server and the game. Through the page's own buttons: «Игра» -> the task, the speed
-> «Запустить игру»; the deployment - «Принять подсказку»; the operative at 0 s «Оставить как есть», at 15 s the hint
taken after holding the pause 5 s, at 30 s «Поменять» and one group edit (a unit clicked and moved to another group by
its key), then «Оставить как есть» to the end. Screenshots game_*.png in --out; a json of what was checked on stdout.

    (one shell)  powershell -NoProfile -ExecutionPolicy Bypass -File tools/label/game.ps1 -Graphics low -NoBrowser -StopAfterSeconds 20 -TimeoutModelSeconds 300
    (another)    python -m tools.label.game_check --task hold --speed 10
"""
import argparse
import json
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from tools.label.browser_check import CHROMES, WS, Page, click_text, screen_of

BANNER = "document.getElementById('banner')"


def wait_pending(page, cond, timeout):
    return page.until(f"S.pending && S.pv && !S.changing && ({cond})", timeout)


def run(url, out, chrome, task, speed, hold_s=5.0, limit_s=1500):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    prof = tempfile.mkdtemp(prefix="label_game_chrome_")
    port = 9334
    proc = subprocess.Popen([chrome, "--headless=new", f"--remote-debugging-port={port}", f"--user-data-dir={prof}",
                             "--window-size=1600,950", "--force-device-scale-factor=1", "--no-first-run",
                             "--no-default-browser-check", "about:blank"], stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    checked = {"task": task, "speed": speed}
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
        page.until("typeof S!=='undefined' && S.cfg && S.pool", 60)
        checked["server_has_game"] = page.js("S.cfg.game")
        click_text(page, "▶ Игра")
        page.js(f"S.gtask={json.dumps(task)}; S.gspeed={int(speed)}; renderSetup(); true")
        page.shot(out / "game_0_tab.png")
        click_text(page, "Запустить игру")
        t0 = time.time()
        # the deployment: the strategist's pause
        wait_pending(page, "S.pending.level==='strategist'", 600)
        checked["deploy_wait_s"] = round(time.time() - t0, 1)
        checked["deploy_groups"] = page.js("S.pending.hint.groups.map(g=>g.members)")
        page.shot(out / "game_1_deploy.png")
        click_text(page, "Принять подсказку", BANNER)
        # the operative at 0 s: keep the strategist's groups
        wait_pending(page, "S.pending.level==='operative' && S.pending.sec===0", 300)
        checked["op0_groups"] = page.js("S.pending.state.groups.map(g=>g.members)")
        page.shot(out / "game_2_op0.png")
        click_text(page, "Оставить как есть", BANNER)
        # 15 s: hold the pause, then the formula's hint
        wait_pending(page, "S.pending.sec>=15", 300)
        sec = page.js("S.pending.sec")
        time.sleep(hold_s)
        checked["held_pause_at_s"] = sec
        click_text(page, "Поменять", BANNER)
        click_text(page, "Принять подсказку")
        page.until(f"!S.pending || S.pending.sec>{sec}", 60)
        # 30 s: one group edit - a unit boxed and moved to another group
        wait_pending(page, f"S.pending.sec>{sec}", 300)
        sec = page.js("S.pending.sec")
        click_text(page, "Поменять", BANNER)
        time.sleep(0.5)
        groups = page.js("S.edit.groups.map(g=>({g:g.g, members:g.members}))")
        big = max(groups, key=lambda g: len(g["members"]))
        other = next((g for g in groups if g["g"] != big["g"]), None)
        unit = big["members"][-1]
        to = other["g"] if other else (big["g"] + 1) % 6
        o = page.js(f"(()=>{{const o=S.drawn.find(o=>o.slot==={unit});return o?[o.x,o.z]:null;}})()")
        page.click(*screen_of(page, o[0], o[1]))          # a click on my unit selects it
        checked["edit_selected"] = page.js("[...S.sel]")
        key = str(to + 1)
        page.ws.call("Input.dispatchKeyEvent", type="keyDown", key=key, text=key)
        page.ws.call("Input.dispatchKeyEvent", type="keyUp", key=key)
        time.sleep(1.5)
        checked["edit"] = {"unit": unit, "from": big["g"], "to": to,
                           "preview": page.js("S.pv && {ok:S.pv.ok, errors:S.pv.errors, groups:(S.pv.groups||[]).map(g=>[g.g,g.members])}")}
        page.shot(out / "game_3_op_edit.png")
        if not (checked["edit"]["preview"] or {}).get("ok"):
            checked["edit_refused_text"] = page.js("document.getElementById('d-err').innerText")
            page.js("S.edit=JSON.parse(JSON.stringify(S.pending.edit)); preview(); true")
            time.sleep(1.2)
        click_text(page, "Дальше ▶")
        page.until(f"!S.pending || S.pending.sec>{sec}", 60)
        checked["after_edit"] = page.js("document.getElementById('status').textContent")
        # to the end: keep
        n = 0
        while time.time() - t0 < limit_s:
            if page.js("!!document.querySelector('#result:not(.hidden)')"):
                break
            if page.js("!!S.pending && !!S.pv && !S.changing"):
                click_text(page, "Оставить как есть" if page.js("S.pending.can_keep") else "Принять подсказку", BANNER)
                n += 1
            time.sleep(0.3)
        checked["keeps_after"] = n
        checked["result"] = page.js("S.st.result")
        checked["folder"] = page.js("S.st.folder")
        page.shot(out / "game_4_result.png")
    except Exception as e:
        checked["error"] = f"{type(e).__name__}: {e}"
        try:
            checked["page"] = page.js("({status: document.getElementById('status').textContent, err: "
                                      "document.getElementById('d-err').innerText, phase: S.st && S.st.phase, "
                                      "error: S.st && S.st.error})")
            page.shot(out / "game_failed.png")
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
    ap.add_argument("--out", default="build/diag_chain/label_game")
    ap.add_argument("--chrome", default=next((c for c in CHROMES if Path(c).exists()), None))
    ap.add_argument("--task", default="hold", choices=("hold", "destroy"))
    ap.add_argument("--speed", type=int, default=10)
    a = ap.parse_args(argv)
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(run(a.url, a.out, a.chrome, a.task, a.speed), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
