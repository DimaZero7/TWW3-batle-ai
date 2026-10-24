"""Who plays whom (docs/en/training/training.md).

Every battle of the batch has a fixed layout for the whole run: its scene, the side the learner
plays and the opponent. Opponents:

    self        the learner on both sides (both sides give training data)
    past        a past version of the learner from the pool (Pool: drawn by its quality score; each battle
                keeps the version it began with: rollout.Battles' past slots)
    nearest, hold_shoot, hold, ai_like   the scripted opponents (tools/nn/train/opponents.py)
    drill_<name>  a drill's enemy script on that drill's battles (tools/nn/train/drills; run.py --drills)

The pool: <pool folder>/*.pt in the checkpoint format (tools/nn/train/checkpoint.py) and q.json (Pool).
"""
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tools import config as project
from tools.nn.train import checkpoint
from tools.nn.train import drills

LEARNER = 0
# (a new opponent goes at the END: the codes of the others stay; "human": opponents.human, not in the mix)
OPPONENTS = (("self", "past", "nearest", "hold_shoot", "hold", "ai_like") + tuple(drills.opponent(n) for n in drills.NAMES)
             + ("human",))
CODE = {name: i + 1 for i, name in enumerate(OPPONENTS)}     # controller codes; 0 = the learner
MIX = {"self": 0.1, "past": 0.15, "nearest": 0.2, "hold_shoot": 0.1, "hold": 0.05, "ai_like": 0.4}


@dataclass
class Layout:
    scene: np.ndarray       # [B] index into the scene list
    learner: np.ndarray     # [B] the learner's side, 1 or 2
    opponent: np.ndarray    # [B] opponent code (CODE)

    @property
    def B(self):
        return len(self.scene)

    def controllers(self):
        """[B, 2] who gives the orders of side 1 and side 2: LEARNER or an opponent code."""
        out = np.zeros((self.B, 2), dtype=np.int64)
        for s in (1, 2):
            mine = self.learner == s
            self_play = self.opponent == CODE["self"]
            out[:, s - 1] = np.where(mine | self_play, LEARNER, self.opponent)
        return out


def with_drills(mix, share, weights=None, names=drills.NAMES):
    """The mix with `share` of the battles given to the drills (by weights {name: w}, default equal; a
    drill of weight 0 is left out), the other opponents scaled to 1 - share."""
    if not share:
        return dict(mix)
    w = {n: float((weights or {}).get(n, 1.0 if weights is None else 0.0)) for n in names}
    w = {n: v for n, v in w.items() if v > 0}
    total = sum(w.values())
    assert total > 0, "--drills needs a drill with a weight above 0"
    base = sum(v for k, v in mix.items() if not k.startswith(drills.PREFIX))
    out = {k: v / base * (1 - share) for k, v in mix.items() if not k.startswith(drills.PREFIX)}
    out.update({drills.opponent(n): share * v / total for n, v in w.items()})
    return out


def counts(B, mix):
    """Battles per opponent, proportional to mix, summing to B."""
    names = [n for n in OPPONENTS if mix.get(n, 0) > 0]
    w = np.array([mix[n] for n in names], float)
    raw = w / w.sum() * B
    n = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - n))[:B - n.sum()]:
        n[i] += 1
    return dict(zip(names, n))


ATTACK_ONLY_DEFAULT = ("hold",)


def opponent_key(name):
    """The log's name of an opponent of run.py --opponents: a script's (or "self") own name; a checkpoint's file name
    without .pt, with its folder's name before a generic one (m40, latest, best, v00012: x2_exploit_general_m40)."""
    if name in CODE:
        return name
    p = Path(name)
    stem = p.stem
    if re.fullmatch(r"(m\d+|latest|best|v\d+|random)", stem) and p.parent.name:
        stem = f"{p.parent.name}_{stem}"
    return stem


def parse_opponents(text):
    """{name: share} of run.py --opponents (a JSON object, or "name:share,name:share"): every opponent by name - a
    script or "self" (league.OPPONENTS, not "past"), or a checkpoint (a path, a past version of its own); the shares
    above 0 and summing to 1 (within 1e-3); the log's names (opponent_key) distinct. ValueError otherwise."""
    text = (text or "").strip()
    if not text:
        raise ValueError("--opponents: empty")
    if text.startswith("{"):
        shares = json.loads(text)
    else:
        shares = {}
        for part in text.split(","):
            name, colon, v = part.strip().rpartition(":")
            if not colon or not name:
                raise ValueError(f"--opponents {part!r}: name:share")
            shares[name.strip()] = float(v)
    shares = {str(k): float(v) for k, v in shares.items()}
    if "past" in shares:
        raise ValueError("--opponents: name the past versions by their checkpoints, not 'past'")
    bad = [k for k, v in shares.items() if not v > 0]
    if bad:
        raise ValueError(f"--opponents: shares must be above 0: {bad}")
    total = sum(shares.values())
    if abs(total - 1.0) > 1e-3:
        raise ValueError(f"--opponents: the shares sum to {total:.4f}, not 1")
    keys = [opponent_key(k) for k in shares]
    if len(set(keys)) != len(keys):
        raise ValueError(f"--opponents: two opponents with one name in the log: {keys}")
    return shares


def is_checkpoint(name):
    """An --opponents name that is a checkpoint (not a script or "self")."""
    return name not in CODE


@dataclass
class Named:
    """The layout of run.py --opponents: the battles' Layout (the checkpoints' rows have the code "past"), each row's
    past slot (row_slot [B], -1: not a checkpoint's), the checkpoints in slot order (paths) and every opponent's log
    name and rows."""
    layout: "Layout"
    row_slot: np.ndarray
    checkpoints: list
    keys: dict          # name -> log name (opponent_key)
    rows: dict          # name -> battles (rows) of the batch


def named_layout(B, n_scenes, shares, scene_attacker=None, attack_only=ATTACK_ONLY_DEFAULT):
    """Named of B battles by the hard shares {name: share} (parse_opponents): each opponent gets its rows by the
    largest remainder (sum B: every share within one battle of share x B), within them the scenes and the learner's
    side cycle as in layout(); a checkpoint's rows are past battles in its own slot for the whole run."""
    names = list(shares)
    w = np.array([shares[n] for n in names], float)
    raw = w / w.sum() * B
    n = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - n), kind="stable")[:B - n.sum()]:
        n[i] += 1
    ckpts = [nm for nm in names if is_checkpoint(nm)]
    scene, side, opp, slot = [], [], [], []
    for name, k in zip(names, n):
        j = np.arange(k)
        sc = j % n_scenes
        sd = 1 + (j // n_scenes) % 2
        if scene_attacker is not None and name in attack_only:
            sd = np.asarray(scene_attacker)[sc]
        scene.append(sc)
        side.append(sd)
        ck = is_checkpoint(name)
        opp.append(np.full(k, CODE["past"] if ck else CODE[name]))
        slot.append(np.full(k, ckpts.index(name) if ck else -1))
    lay = Layout(np.concatenate(scene), np.concatenate(side), np.concatenate(opp))
    return Named(lay, np.concatenate(slot), ckpts, {nm: opponent_key(nm) for nm in names},
                 {nm: int(k) for nm, k in zip(names, n)})


ATTACK_ONLY = ("hold",)   # `hold` never attacks: as the attacker it only waits out the hour
# (the evaluation's set; training takes its own from run.py --defend-only, default this one)


def names_list(text):
    """('a', 'b') of 'a,b' (run.py --defend-only); every name must be an opponent."""
    names = tuple(n.strip() for n in (text or "").split(",") if n.strip())
    bad = [n for n in names if n not in CODE]
    if bad:
        raise ValueError(f"not opponents: {bad} (league.OPPONENTS)")
    return names


def layout(B, n_scenes, mix=None, opponent=None, scene_attacker=None, attack_only=ATTACK_ONLY):
    """A layout of B battles: within each opponent's share the scenes and the learner's side
    cycle, so every opponent meets every (scene, side) about equally. opponent: one name for all
    battles (evaluation). scene_attacker [n_scenes] (1 or 2): with it, the opponents in attack_only
    play only the defender (the learner takes the attacker's side)."""
    per = {opponent: B} if opponent else counts(B, mix or MIX)
    scene, side, opp = [], [], []
    for name, n in per.items():
        j = np.arange(n)
        sc = j % n_scenes
        sd = 1 + (j // n_scenes) % 2
        if scene_attacker is not None and name in attack_only:
            sd = np.asarray(scene_attacker)[sc]
        scene.append(sc)
        side.append(sd)
        opp.append(np.full(n, CODE[name]))
    return Layout(np.concatenate(scene), np.concatenate(side), np.concatenate(opp))


def inherit_pool(root, sources):
    """A chain step's new pool carries on from the previous step's (run.py; --pool-fresh: not): the first of sources
    (checkpoints: --init, --critic-init) whose folder has pool/q.json - a run's latest.pt beside its pool - is copied
    into root (its versions' files stay where they are: q.json keeps their paths). Only into a folder without its own
    q.json. -> the q.json copied, or None."""
    import shutil
    root = Path(root)
    if (root / Pool.STATE).exists():
        return None
    for src in sources:
        if not src:
            continue
        q = Path(src).parent / "pool" / Pool.STATE
        if q.is_file() and q.parent.resolve() != root.resolve():
            root.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(q, root / Pool.STATE)
            return q
    return None


def open_pool(root, size=8, seed=0, eta=0.01, init=None, critic=None, untrained=None, inherit=True,
              no_untrained=False, extra=()):
    """(Pool, notes) of a run (run.py): root's q.json if there (--pool-dir: the parts of a night), else the previous
    step's (inherit_pool from init / critic, inherit=True), else a new pool with the untrained network
    (untrained(): its file; not with no_untrained while there is an init) and init; no_untrained also takes an
    inherited untrained one out; the extra checkpoints go in (one already in keeps its q)."""
    notes = []
    if inherit:
        q = inherit_pool(root, (init, critic))
        if q is not None:
            notes.append(f"pool: carries on from {Pool.key(q)}")
    pool = Pool(root, size=size, seed=seed, eta=eta)
    if no_untrained and pool.drop_untrained():
        notes.append("pool: the untrained network taken out (--pool-no-untrained)")
    if not len(pool):
        if not (no_untrained and init) and untrained is not None:
            pool.add(untrained(), untrained=True)
        if init:
            pool.add(init)
    for e in extra:
        pool.add(e)
    pool.write()
    return pool, notes


class Pool:
    """Past versions of the learner with quality scores, as OpenAI Five's (arXiv 1912.06680, appendix N): a version
    is drawn with probability p_i ~ exp(q_i); a new one comes in with the highest q of the pool; every battle the
    learner wins against version i lowers q_i by eta / (N p_i) (N: the pool's size, p_i: its probability when it was
    drawn; a loss changes nothing). Beaten versions fade out, those that still win stay likely; the untrained
    network is an ordinary entry (no lasting place: it is beaten and fades out). Over the size the lowest q leaves
    the list (its file stays).

    The pool's state (versions, q, the clock in updates, games) is root/q.json: a pool folder shared by the parts of
    a night (run.py --pool-dir) carries on where the last part left it. Paths inside the project are stored
    relative to it (the container mounts the project elsewhere). The draws come from the seed, the folder (key) and
    the state: two new pools in different folders draw differently, the same folder the same."""

    STATE = "q.json"

    def __init__(self, root=checkpoint.POOL, size=8, seed=0, eta=0.01):
        self.root = Path(root)
        self.size = size
        self.eta = eta
        self.entries = []        # {"path", "q", "added" (clock), "games", "wins", "untrained"}
        self.clock = 0           # updates trained with this pool, over all its runs
        state = self.root / self.STATE
        if state.exists():
            data = json.loads(state.read_text(encoding="utf-8"))
            self.entries = [dict(e) for e in data["entries"]]
            self.clock = int(data.get("clock", 0))
        # the draws from the seed, the pool's folder and its state: every step of a chain has a new folder and draws
        # its own sequence (from "seed-size-clock" alone every new pool drew the same one: the same versions at the
        # same updates, and two of the three extras never played); the same folder draws the same again
        self.rng = random.Random(f"{seed}-{self.key(self.root)}-{len(self.entries)}-{self.clock}")

    def __len__(self):
        return len(self.entries)

    @staticmethod
    def key(path):
        """A path as stored: relative to the project when inside it, with forward slashes."""
        p = Path(path)
        try:
            p = p.resolve().relative_to(project.ROOT)
        except ValueError:
            pass
        return p.as_posix()

    @staticmethod
    def file(key):
        p = Path(key)
        return p if p.is_absolute() else project.ROOT / p

    @property
    def paths(self):
        return [self.file(e["path"]) for e in self.entries]

    def find(self, path):
        k = self.key(path)
        return next((e for e in self.entries if e["path"] == k), None)

    def add(self, path, untrained=False):
        """Add a version with q = the pool's highest (0 in an empty pool); one already in keeps its q."""
        old = self.find(path)
        if old is not None:
            return old
        e = {"path": self.key(path), "q": max((x["q"] for x in self.entries), default=0.0), "added": self.clock,
             "games": 0, "wins": 0, "untrained": bool(untrained)}
        self.entries.append(e)
        while len(self.entries) > self.size:
            self.entries.remove(min(self.entries[:-1], key=lambda x: x["q"]))
        return e

    def save(self, actor, critic=None, preset="small", meta=None, name=None):
        path = self.root / (name or f"v{int((meta or {}).get('update', 0)):05d}.pt")
        checkpoint.save(path, actor, critic, preset, meta)
        self.add(path)
        self.write()
        return path

    def probs(self):
        q = np.array([e["q"] for e in self.entries], float)
        w = np.exp(q - q.max())
        return w / w.sum()

    def sample(self):
        """(path, is the untrained one, its probability now)."""
        p = self.probs()
        i = self.rng.choices(range(len(self.entries)), weights=p)[0]
        e = self.entries[i]
        return self.file(e["path"]), bool(e["untrained"]), float(p[i])

    def result(self, path, games, wins, p):
        """games battles against the version `path` (drawn with probability p), the learner won `wins` of them:
        q -= eta * wins / (N p). A version no longer in the pool is left alone."""
        e = self.find(path)
        if e is None or not games:
            return
        e["games"] += int(games)
        e["wins"] += int(wins)
        e["q"] = float(e["q"] - self.eta * float(wins) / (len(self.entries) * max(float(p), 1e-9)))

    def tick(self):
        self.clock += 1

    def write(self):
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.root / (self.STATE + ".tmp")
        tmp.write_text(json.dumps({"clock": self.clock, "eta": self.eta, "entries": self.entries}, indent=1),
                       encoding="utf-8", newline="\n")
        os.replace(tmp, self.root / self.STATE)

    def drop_untrained(self):
        """Take the untrained network out (run.py --pool-no-untrained); -> how many left."""
        n = len(self.entries)
        self.entries = [e for e in self.entries if not e.get("untrained")]
        return n - len(self.entries)

    def top(self, k=5):
        """[(name, q, p, age in updates)] of the k most probable versions."""
        p = self.probs()
        order = np.argsort(-p)[:k]
        return [(Path(self.entries[i]["path"]).name, self.entries[i]["q"], float(p[i]),
                 self.clock - self.entries[i]["added"]) for i in order]

    def text(self, k=5):
        """One line: the pool's size, versions above 5 % (a steep distribution: the learner outgrows its past fast;
        a wide one: it stalls), the top k with q, p and age."""
        p = self.probs()
        return (f"pool {len(self)} versions (clock {self.clock}), p > 5 %: {int((p > 0.05).sum())}; top "
                + ", ".join(f"{n} q {q:+.2f} p {pp:.2f} age {a}" for n, q, pp, a in self.top(k)))
