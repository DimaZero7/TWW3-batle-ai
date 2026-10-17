"""Checks of the map analysis "now and then" (design/general/index.html, "Проверки анализа карты", group 2): run
after training when there is a reason (the network acts oddly with the map, before widening the features, after
a big rework), not every time. Functions only; nothing runs them by itself.

    dead(model, maps)          features that never change on any map (a dead ReLU: always 0)
    crowding(model, maps)      effective rank: how many of the features are really different ("тесно")
    twins(model, maps)         similarity table of the features, pairs that repeat each other
    feature_maps(model, layers) -> features_png(maps, path)   picture of every feature of one map
    similarity(model, maps) -> similarity_png(table, path)

maps: [N, C, 128, 128] map layers (map_input.to_tensor, map_batch.layers); the more different maps the better.
The statistics run over every map and every position of the step's grid.
"""
import math

try:
    import torch
except ImportError:          # the host draws pictures (features_png, similarity_png) without torch
    torch = None

STEP_NAMES = ("stem", "conv1", "pool1", "conv2", "pool2", "head")


def _samples(model, maps, step=-1, chunk=64):
    """[N * h * w, F] the step's features over every map and position."""
    out = []
    with torch.no_grad():
        for i in range(0, maps.shape[0], chunk):
            f = model.steps(maps[i:i + chunk])[step]
            out.append(f.permute(0, 2, 3, 1).reshape(-1, f.shape[1]).double())
    return torch.cat(out)


def dead(model, maps, step=-1, tol=1e-6):
    """{"dead": [feature numbers that never change], "share": their share, "zero": [always exactly 0]}."""
    x = _samples(model, maps, step)
    spread = x.max(0).values - x.min(0).values
    d = [int(k) for k in torch.nonzero(spread <= tol).flatten()]
    z = [int(k) for k in torch.nonzero(x.abs().max(0).values == 0).flatten()]
    return {"dead": d, "share": len(d) / x.shape[1], "zero": z}


def dead_all(model, maps, tol=1e-6):
    """dead() share at every step: {step name: share}."""
    return {n: dead(model, maps, k, tol)["share"] for k, n in enumerate(STEP_NAMES)}


def crowding(model, maps, step=-1):
    """Effective rank of the features (exp of the entropy of the normalised singular values of the centred,
    standardised samples; Roy & Vetterli 2007) and its share of the feature count. Share near 1: every feature
    is busy and different - room is tight, widen to 64. Also how many components hold 90 / 99 % of the
    variance."""
    x = _samples(model, maps, step)
    x = x - x.mean(0)
    sd = x.std(0)
    live = sd > 1e-9
    x = x[:, live] / sd[live]
    s = torch.linalg.svdvals(x)
    p = s / s.sum()
    erank = float(torch.exp(-(p * torch.log(p.clamp(min=1e-30))).sum()))
    var = s ** 2 / (s ** 2).sum()
    cum = torch.cumsum(var, 0)
    n = model.features
    return {"features": n, "live": int(live.sum()), "effective_rank": erank, "share": erank / n,
            "for_90": int((cum < 0.90).sum()) + 1, "for_99": int((cum < 0.99).sum()) + 1}


def similarity(model, maps, step=-1):
    """[F, F] correlation of the features over every map and position (0 for a feature that never changes)."""
    x = _samples(model, maps, step)
    x = x - x.mean(0)
    sd = x.std(0)
    x = x / sd.clamp(min=1e-12)
    c = (x.T @ x) / (x.shape[0] - 1)
    c[sd <= 1e-9] = 0
    c[:, sd <= 1e-9] = 0
    return c


def twins(model, maps, step=-1, limit=0.95):
    """Pairs of features whose |correlation| >= limit: [(i, j, corr)], strongest first; and the table."""
    c = similarity(model, maps, step)
    n = c.shape[0]
    pairs = [(i, j, float(c[i, j])) for i in range(n) for j in range(i + 1, n) if abs(float(c[i, j])) >= limit]
    return sorted(pairs, key=lambda p: -abs(p[2])), c


def similarity_png(c, path, step=-1):
    """Picture of a similarity() table (numpy or torch)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = c.numpy() if hasattr(c, "numpy") else c
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax)
    ax.set_title(f"feature similarity, step {STEP_NAMES[step]}")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return path


def feature_maps(model, layers, step=-1):
    """[F, h, w] numpy: every feature of one map ([C, 128, 128] layers) at the step."""
    with torch.no_grad():
        return model.steps(layers[None])[step][0].double().cpu().numpy()


def features_png(f, path, step=-1, title=""):
    """Picture of feature_maps() (numpy [F, h, w]), a panel each (row 0 = low z, as render.layers_png). The
    training image has no matplotlib: save feature_maps() with numpy there, draw on the host."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = f.shape[0]
    cols = 8
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.2 * rows))
    for k, ax in enumerate(axes.flat):
        ax.set_xticks([])
        ax.set_yticks([])
        if k >= n:
            ax.axis("off")
            continue
        ax.imshow(f[k], origin="lower", cmap="viridis", interpolation="nearest")
        ax.set_title(f"{k}: {f[k].min():.2g}..{f[k].max():.2g}", fontsize=7)
    fig.suptitle(f"{title} step {STEP_NAMES[step]}: {n} features {f.shape[1]} x {f.shape[2]}")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return path
