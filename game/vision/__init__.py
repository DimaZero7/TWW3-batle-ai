"""The orchestrator's eyes: a battle (a game recording or the simulator's frames) -> a strip of frames from above (PNG)
and a text of events and measures, ~3 k tokens a battle (game/vision/README.md).

    record.py   a recording / the simulator's frames -> Battle (frames by second, roles, the view's axes, groups)
    events.py   contact, routs, fast enemies on our shooters, holes in the line, orders a minute, the strip's moments
    draw.py     the strip (ground, units in their real size and facing, melee, rout, fire, groups: decided vs done)
    text.py     the measures and events table (~1.6-2 k tokens); ascii_frames - the character map (not recommended)
"""
from pathlib import Path


def render(battle, out_dir, frames=6, size=360, cols=3, png=True, txt=True):
    """Writes <out_dir>/<battle>.png and .txt; returns {png, txt, tokens: {image, text}}."""
    from game.vision import draw, events, text
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fx = events.Facts(battle)
    res = {"tokens": {"image": 0, "text": 0}}
    if png:
        p = out_dir / f"{battle.name}.png"
        im = draw.strip(fx, events.pick_frames(fx, n=frames), size=size, cols=cols, path=p)
        res["png"], res["tokens"]["image"] = str(p), draw.tokens(*im.size)
    if txt:
        p = out_dir / f"{battle.name}.txt"
        t = text.table(fx)
        p.write_text(t, encoding="utf-8")
        res["txt"], res["tokens"]["text"] = str(p), text.tokens(t)
    return res
