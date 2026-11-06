"""The labelling server: the page (tools/label/web) and its JSON API, one battle at a time (stdlib http.server).

    bash tools/label/serve.sh            (the container: torch; then http://localhost:8790)
    python -m tools.label.server [--port 8790] [--host 0.0.0.0] [--device cpu]

GET  /api/pool          factions and their units (pool.factions), maps (pool.maps), the deployment strips
GET  /api/config        the levels with their periods, the enemy's modes, the verbs and paces, the group colours
GET  /api/matchups      the saved matchups' names;  GET /api/matchup?name=...   one
POST /api/matchup       save (checked: matchup.check)
POST /api/check         a matchup's errors (nothing saved)
POST /api/start         {matchup, level, period, upper, lower, name} -> a new battle (the old one ends)
POST /api/advance       {seconds: n | null}: on until the person's decision / the end / n seconds -> {snaps, state}
GET  /api/state
POST /api/preview       {edit}: the decision checked and played by the walk, nothing changed
POST /api/decide        {edit}: the same, written and put in force
POST /api/stop          the end now
GET  /maps/<name>/preview.png   the captured map's picture
"""
import argparse
import json
import mimetypes
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from tools import config as project
from tools.label import cfg
from tools.label import matchup as MU
from tools.label import pool as PL

WEB = Path(__file__).parent / "web"
LOCK = threading.Lock()
STATE = {"session": None, "nets": {}, "device": "cpu"}


def nets(kind):
    """The networks, loaded once: unit (always), operative (run5), tactic (run4)."""
    have = STATE["nets"]
    if kind not in have:
        dev = STATE["device"]
        from tools.nn.tactic import train as TT
        if kind == "unit":
            have[kind] = TT.unit_network(dev)
        elif kind == "operative":
            from tools.nn.operative import train as OT
            net, _, _ = OT.restore(str(project.ROOT / cfg()["nets"]["operative"]), dev)
            have[kind] = net.eval()
        elif kind == "tactic":
            from tools.nn.operative import cfg as ocfg
            net, _, _ = TT.restore(str(project.ROOT / ocfg()["tactic_ckpt"]), dev)
            have[kind] = net.eval()
    return have[kind]


def strategist_ready():
    """None when the strategist's level can run (its code imports), else why not."""
    try:
        from tools.label import strategist  # noqa: F401
        return None
    except Exception as e:                       # (the strategist's code is another agent's, still being written)
        return f"стратег не подключается: {type(e).__name__}: {e}"


def config_info():
    from tools.nn.operative import cfg as ocfg
    from tools.nn.tactic import cfg as tcfg
    c = cfg()
    return {"levels": [{"name": "operative", "label": c["levels"]["operative"]["label"], "period": ocfg()["decide_s"],
                        "ready": True},
                       {"name": "tactic", "label": c["levels"]["tactic"]["label"], "period": tcfg()["decide_s"],
                        "ready": True},
                       {"name": "strategist", "label": c["levels"]["strategist"]["label"], "period": None,
                        "ready": strategist_ready() is None, "why": strategist_ready()}],
            "enemy_modes": c["enemy_modes"], "verbs": list(MU.VERBS), "paces": list(MU.PACES),
            "colors": c["group_colors"], "limit_s": c["limit_s"], "max_per_side": MU.MAX_PER_SIDE}


def pool_info():
    from tools.label import mapimg
    maps = []
    for m in PL.maps():
        d = dict(m, **PL.field(m["name"]))
        d["extent"] = mapimg.picture(m["name"])[1] if m["name"] != "flat" else None
        maps.append(d)
    return {"factions": PL.factions(), "maps": maps}


def start(body):
    from tools.label import session as SE
    old = STATE["session"]
    if old is not None and not old.over:
        old.stop()
    level = body.get("level", "operative")
    upper, lower = body.get("upper", "formula"), body.get("lower", "formula")
    n = {"unit": nets("unit")}
    if (level == "tactic" and upper == "net") or (level == "strategist" and lower == "net"):
        n["operative"] = nets("operative")
    if level == "operative" and lower == "net":
        n["tactic"] = nets("tactic")
    period = body.get("period")
    period = int(period) if period else None
    s = SE.Session(body["matchup"], level=level, period=period, upper=upper, lower=lower, nets=n,
                   device=STATE["device"], name=body.get("name") or body["matchup"].get("name"))
    STATE["session"] = s
    return s.state()


def api(method, path, q, body):
    s = STATE["session"]
    if path == "/api/pool":
        return pool_info()
    if path == "/api/config":
        return config_info()
    if path == "/api/matchups":
        return {"names": MU.names()}
    if path == "/api/matchup" and method == "GET":
        return MU.load(q["name"][0])
    if path == "/api/matchup":
        return {"name": MU.save(body)}
    if path == "/api/check":
        return {"errors": MU.check(body)}
    if path == "/api/start":
        return start(body)
    if s is None:
        raise ValueError("боя нет: начни его (/api/start)")
    if path == "/api/state":
        return s.state()
    if path == "/api/advance":
        from tools.label import view as VW
        snaps = s.advance(body.get("seconds"))
        return {"snaps": [VW.for_page(x) for x in snaps], "state": s.state()}
    if path == "/api/preview":
        return s.preview(body.get("edit") or {})
    if path == "/api/decide":
        out = s.decide(body.get("edit") or {})
        out["state"] = s.state()
        return out
    if path == "/api/stop":
        s.stop()
        return s.state()
    raise KeyError(path)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, code, data, ctype="application/json; charset=utf-8"):
        if not isinstance(data, (bytes, bytearray)):
            data = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _static(self, path):
        if path.startswith("/mapimg/") and path.endswith(".png"):
            from tools.label import mapimg
            name = path[len("/mapimg/"):-4]
            if MU.safe(name) == name and (PL.MAPS / name / "heights.npz").exists():
                return self._send(200, mapimg.picture(name)[0], "image/png")
            return self._send(404, {"error": "no such map"})
        name = "index.html" if path in ("", "/") else path.lstrip("/")
        f = (WEB / name).resolve()
        if WEB.resolve() not in f.parents or not f.is_file():
            return self._send(404, {"error": "not found"})
        ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or f.suffix == ".js":
            ctype = ("text/javascript" if f.suffix == ".js" else ctype) + "; charset=utf-8"
        return self._send(200, f.read_bytes(), ctype)

    def _handle(self, method):
        u = urlparse(self.path)
        if not u.path.startswith("/api/"):
            return self._static(u.path)
        body = {}
        if method == "POST":
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n).decode("utf-8") or "{}") if n else {}
        try:
            with LOCK:
                out = api(method, u.path, parse_qs(u.query), body)
            return self._send(200, out)
        except Exception as e:                                  # the page shows the message
            errs = getattr(e, "errors", None) or [f"{type(e).__name__}: {e}"]
            if not hasattr(e, "errors"):
                traceback.print_exc()
            return self._send(400, {"ok": False, "errors": errs})

    def do_GET(self):
        self._handle("GET")

    def do_POST(self):
        self._handle("POST")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=int(cfg()["port"]))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    STATE["device"] = a.device
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"Разметка: http://localhost:{a.port}  (Ctrl+C - стоп)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        s = STATE["session"]
        if s is not None and not s.over:
            s.stop()
        srv.server_close()


if __name__ == "__main__":
    main()
