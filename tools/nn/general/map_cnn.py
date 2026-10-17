"""The general's map analysis, "Анализ карты" (design/general/index.html): a convolutional network that reads the
map's layers (map_input.py: [6, 128, 128], always as the side the network plays sees it) and gives its summary:
1024 regions of 64 m (32 x 32), 32 numbers each -> tokens [B, 1024, 32] for the general's "meeting".

    stem   3x3 window over the input layers -> F features     128 x 128   (+ ReLU)
    conv   3x3, F -> F                                        128 x 128   (+ ReLU)
    pool   2 x 2 -> the largest                                64 x  64
    conv   3x3, F -> F                                         64 x  64   (+ ReLU)
    pool   2 x 2 (or 3 x 3: regions of 96 m)                   32 x  32
    head   3x3, F -> F (no activation: the tokens)             32 x  32
F = 32: 29 504 weights (F = 64: 114 304). The last region reaches 4 x 16 m = 64 m (head, pool, conv, pool,
conv, stem: 8 cells = 128 m) beyond its own cells each way: it sees 320 x 320 m.

Beyond the grid is "outside", not zero. The input gets a margin of "outside" cells (field 0, outside 1, flags 0)
as wide as the network reaches (8 cells for regions of 64 m) and every window runs only where it sees real
cells ("valid", no padding inside the network): the answer in the grid is the answer of an endless plane of
"outside" around the map, so a cell at the grid's edge is read exactly as one in the middle of a map surrounded
by "outside" - the grid edge is invisible (test_the_grid_edge_reads_as_the_middle). The margin also lines the
pools up with the regions: region r holds the cells [r * S, (r + 1) * S), S = 4 (6 for 96 m).

New input layers (forest, water, height...) join the first window with zero weights (add_layers): a separate
window over the new layers, without bias, starting at 0, added to the first window's answer. Its output is exactly
0, so the old network answers exactly as before (to the bit) until training moves the new weights.

The summary is computed once per map (the terrain does not change in a battle): MapCNN.indexed() for a batch of
battles sharing maps (each distinct map once, gradients flow), SummaryCache for rollouts (no gradients) by any key
(a map's number and side, a drill battle's number), dropped whenever the weights change.

    python -m tools.nn.general.map_cnn --bench            weights, time and memory: 1 map, 1024 drill battles
"""
import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.nn.general import map_input as mi

# The input layers' values beyond the grid (map_input.LAYERS): nothing but "outside".
OUTSIDE = (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)


class MapCNN(nn.Module):
    """Map layers [B, C, 128, 128] -> tokens [B, R*R, features] (module doc).

    features: the number of features (32; 64 when terrain comes); pools: the two poolings' sizes, (2, 2) ->
    32 x 32 regions of 64 m, (2, 3) -> 22 x 22 regions of 96 m (the last row and column reach past the grid:
    "outside"); outside: the input layers' values beyond the grid."""

    def __init__(self, features=32, pools=(2, 2), outside=OUTSIDE):
        super().__init__()
        f = int(features)
        self.features = f
        self.pools = tuple(int(p) for p in pools)
        p1, p2 = self.pools
        self.regions = math.ceil(mi.GRID / (p1 * p2))
        # margins (cells) of "outside" before / after the grid: valid windows and floor pools end with exactly
        # `regions` regions, region 0 starting at cell 0 (module doc)
        self.margin_lo = 2 + p1 * (1 + p2)
        self.margin_hi = ((self.regions + 2) * p2 + 2) * p1 + 4 - mi.GRID - self.margin_lo
        self.stem = nn.ModuleList([nn.Conv2d(len(outside), f, 3)])
        self.stem_outside = [tuple(float(v) for v in outside)]
        self.convs = nn.ModuleList([nn.Conv2d(f, f, 3) for _ in range(3)])
        for c in list(self.stem) + list(self.convs):
            # He init (ReLU): the numbers keep their scale from step to step (test_numbers_keep_their_scale)
            nn.init.kaiming_normal_(c.weight, nonlinearity="relu")
            nn.init.zeros_(c.bias)

    # --- input layers ---------------------------------------------------------------------------------------
    @property
    def in_layers(self):
        return sum(c.in_channels for c in self.stem)

    def add_layers(self, n, outside=None):
        """n new input layers (appended after the old ones), joined to the first window with zero weights;
        outside: their values beyond the grid (default 0). The answer stays the same to the bit."""
        outside = tuple(float(v) for v in (outside if outside is not None else (0.0,) * n))
        assert len(outside) == n
        w = self.stem[0].weight
        c = nn.Conv2d(n, self.features, 3, bias=False).to(device=w.device, dtype=w.dtype)
        nn.init.zeros_(c.weight)
        self.stem.append(c)
        self.stem_outside.append(outside)
        return self

    def _framed(self, x, value):
        """x [B, C, n, n] inside a margin of value [C] ("outside")."""
        b, c, h, w = x.shape
        lo, hi = self.margin_lo, self.margin_hi
        v = torch.tensor(value, device=x.device, dtype=x.dtype).view(1, c, 1, 1)
        out = v.expand(b, c, h + lo + hi, w + lo + hi).clone()
        out[:, :, lo:lo + h, lo:lo + w] = x
        return out

    # --- forward ------------------------------------------------------------------------------------------
    def raw_steps(self, x):
        """Every step's feature maps with the margin: [stem, conv1, pool1, conv2, pool2, head]."""
        assert x.shape[1] == self.in_layers, f"{x.shape[1]} input layers, the network reads {self.in_layers}"
        at = 0
        h = None
        for c, out in zip(self.stem, self.stem_outside):
            part = self._framed(x[:, at:at + c.in_channels], out)
            at += c.in_channels
            h = c(part) if h is None else h + c(part)
        res = [F.relu(h)]
        res.append(F.relu(self.convs[0](res[-1])))
        res.append(F.max_pool2d(res[-1], self.pools[0]))
        res.append(F.relu(self.convs[1](res[-1])))
        res.append(F.max_pool2d(res[-1], self.pools[1]))
        res.append(self.convs[2](res[-1]))
        return res

    def crops(self):
        """(start, size) of the grid's part of each step of raw_steps."""
        p1 = self.pools[0]
        lo = self.margin_lo
        n1 = math.ceil(mi.GRID / p1)
        return [(lo - 1, mi.GRID), (lo - 2, mi.GRID), ((lo - 2) // p1, n1), ((lo - 2) // p1 - 1, n1),
                (1, self.regions), (0, self.regions)]

    def steps(self, x):
        """Every step's feature maps over the grid only: stem and conv1 [B, F, 128, 128], pool1 and conv2
        [.., 64, 64], pool2 and head [.., R, R] (cell / region k of each covers the same ground as the grid's)."""
        return [f[..., s:s + n, s:s + n] for f, (s, n) in zip(self.raw_steps(x), self.crops())]

    def grid(self, x):
        """Map layers [B, C, 128, 128] -> the summary as a grid [B, F, R, R]."""
        g = self.raw_steps(x)[-1]
        assert g.shape[-1] == self.regions
        return g

    def forward(self, x):
        """Map layers [B, C, 128, 128] -> tokens [B, R*R, F] (row-major: region index = iz * R + ix)."""
        return self.grid(x).flatten(2).transpose(1, 2)

    def indexed(self, maps, index):
        """Tokens of battles that share maps: maps [M, C, 128, 128] (each distinct map once), index [B] the map
        of each battle -> [B, R*R, F]; each map is computed once, gradients flow."""
        return self(maps)[index]

    def region_m(self):
        return mi.CELL_M * self.pools[0] * self.pools[1]


def fingerprint(model):
    """Changes whenever any weight changes in place (an optimiser step, load_state_dict) or is replaced."""
    return tuple((id(p), p._version) for p in model.parameters())


class SummaryCache:
    """Tokens [R*R, F] per key (a map's (number, side), a drill battle's number...), computed without gradients
    once, dropped when the model's weights change (fingerprint)."""

    def __init__(self, model, max_items=None):
        self.model = model
        self.max_items = max_items
        self._store = {}
        self._fp = None
        self.computed = 0

    def _check(self):
        fp = fingerprint(self.model)
        if fp != self._fp:
            self._store.clear()
            self._fp = fp

    def clear(self):
        self._store.clear()

    def __len__(self):
        return len(self._store)

    def get(self, keys, make_layers, chunk=64):
        """keys [B] hashable; make_layers(missing keys) -> [M, C, 128, 128] for the keys not yet held.
        -> [B, R*R, F] (no gradients)."""
        self._check()
        missing = list(dict.fromkeys(k for k in keys if k not in self._store))
        if missing:
            with torch.no_grad():
                for i in range(0, len(missing), chunk):
                    part = missing[i:i + chunk]
                    tok = self.model(make_layers(part))
                    for k, t in zip(part, tok):
                        self._store[k] = t
                    self.computed += len(part)
            if self.max_items and len(self._store) > self.max_items:
                for k in list(self._store)[:len(self._store) - self.max_items]:
                    if k not in keys:
                        del self._store[k]
        return torch.stack([self._store[k] for k in keys])


def map_layers(data_by_number, device="cpu"):
    """make_layers for SummaryCache of stored maps: data_by_number {map number: map_input.load() data},
    keys (map number, side)."""
    def make(keys):
        return torch.stack([mi.to_tensor(data_by_number[n], side, device) for n, side in keys])
    return make


def weights(model):
    return sum(p.numel() for p in model.parameters())


# --- the measurement ------------------------------------------------------------------------------------

def _timed(fn, dev, reps):
    """(seconds per call, peak extra memory MB: cuda allocator's; on cpu the process's peak resident size
    growth, a rough figure - a later phase only shows above an earlier one's peak)."""
    import resource
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if dev.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
    fn()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / reps
    if dev.type == "cuda":
        mem = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
    else:
        mem = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - rss0) / 1024
    return dt, mem


def bench(device=None, features=32, pools=(2, 2), battles=1024, chunk=256):
    from tools.nn.general import map_batch as mb
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(0)
    model = MapCNN(features, pools).to(dev)
    print(f"features {features}, pools {pools}: regions {model.regions} x {model.regions} of "
          f"{model.region_m():g} m, weights {weights(model)}; device {dev}")
    arena = mi.to_tensor(mi.build(mi.arena_spec()), 1, dev)[None]
    with torch.no_grad():
        dt, mem = _timed(lambda: model(arena), dev, 20)
    print(f"1 map, no grad: {dt * 1e3:.2f} ms, peak extra memory {mem:.1f} MB")

    rects = mb.rects_tensor([mb.zones_of(_drill_desc(i)) for i in range(battles)], dev)
    sides = torch.tensor([1 + i % 2 for i in range(battles)], device=dev)
    dt, _ = _timed(lambda: mb.layers(rects, sides), dev, 3)
    print(f"{battles} drill battles, layers (map_batch): {dt * 1e3:.1f} ms")
    lay = mb.layers(rects, sides)

    def nograd():
        with torch.no_grad():
            return torch.cat([model(lay[i:i + chunk]) for i in range(0, battles, chunk)])
    dt, mem = _timed(nograd, dev, 3)
    print(f"{battles} battles, no grad (chunks of {chunk}): {dt * 1e3:.1f} ms, peak extra {mem:.0f} MB")

    with torch.no_grad():
        st = model.raw_steps(lay[:1])
    act = sum(f.numel() for f in st) * 4 / 2 ** 20
    print(f"activations of one map (all steps, float32): {act:.1f} MB; training keeps about twice (ReLU, pool)")
    for n in (64, 256):
        x = lay[:n]

        def train():
            model.zero_grad(set_to_none=True)
            model(x).square().mean().backward()
        dt, mem = _timed(train, dev, 3)
        print(f"{n} distinct maps, forward + backward: {dt * 1e3:.1f} ms, peak extra {mem:.0f} MB "
              f"({mem / n:.1f} MB per map)")


def _drill_desc(i):
    from tools.nn.train import drills
    ds = getattr(_drill_desc, "pool", None)
    if ds is None:
        ds = []
        for d in drills.load().values():
            ds += [desc for desc, _ in drills.battles(d, range(8))]
        _drill_desc.pool = ds
    return ds[i % len(ds)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--features", type=int, default=32)
    ap.add_argument("--pools", default="2,2")
    ap.add_argument("--battles", type=int, default=1024)
    a = ap.parse_args(argv)
    if a.bench:
        bench(a.device, a.features, tuple(int(p) for p in a.pools.split(",")), a.battles)


if __name__ == "__main__":
    main()
