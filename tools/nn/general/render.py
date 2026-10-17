"""Check picture of a map's 6 layers (tools/nn/general/map_input.py): one panel per layer as side 1 sees it,
the true shapes drawn over (field edge, zones, the band around each zone) in world metres."""
import numpy as np

from tools.nn.general import map_input as mi


def _band_outline(poly, r, step_deg=5.0):
    """Points at distance r around a convex polygon (its offset outline), for drawing only."""
    p = np.asarray(poly, float)
    out = []
    n = len(p)
    for i in range(n):
        a, b, c = p[i - 1], p[i], p[(i + 1) % n]
        n1 = np.array([(b - a)[1], -(b - a)[0]]) / max(np.hypot(*(b - a)), 1e-9)
        n2 = np.array([(c - b)[1], -(c - b)[0]]) / max(np.hypot(*(c - b)), 1e-9)
        a1, a2 = np.arctan2(n1[1], n1[0]), np.arctan2(n2[1], n2[0])
        if a2 < a1:
            a2 += 2 * np.pi
        for t in np.arange(a1, a2 + 1e-9, np.radians(step_deg)):
            out.append(b + r * np.array([np.cos(t), np.sin(t)]))
        out.append(b + r * n2)
    return np.asarray(out)


def layers_png(data, spec, path, side=1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lay = mi.unpack(data, side)
    x0, z0 = data["meta"]["origin"]
    ext = (x0, x0 + mi.SIZE_M, z0, z0 + mi.SIZE_M)
    titles = ("1 field (share)", "2 outside (share)", "3 our zone", "4 enemy zone", "5 our vanguard may stand",
              "6 enemy vanguard may stand")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10.5))
    sides = {"our": spec.zones[side], "enemy": spec.zones[3 - side]}
    for k, ax in enumerate(axes.flat):
        ax.imshow(lay[k], origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
        if spec.field_box is not None:
            fx0, fx1, fz0, fz1 = spec.field_box
            ax.plot([fx0, fx1, fx1, fx0, fx0], [fz0, fz0, fz1, fz1, fz0], "w--", lw=1)
        for who, poly in sides.items():
            p = np.vstack([poly, poly[:1]])
            ax.plot(p[:, 0], p[:, 1], "-", color="lime" if who == "our" else "red", lw=1)
            b = _band_outline(poly, spec.band_m)
            b = np.vstack([b, b[:1]])
            ax.plot(b[:, 0], b[:, 1], ":", color="lime" if who == "our" else "red", lw=1)
        on = int((lay[k] > 0.5).sum()) if k >= 2 else float(lay[k].sum())
        ax.set_title(f"{titles[k]} — {'cells yes' if k >= 2 else 'sum'} {on if k >= 2 else round(on, 1)}")
        ax.set_xlabel("x, m")
        ax.set_ylabel("z, m")
    fig.suptitle(f"{spec.name}: {mi.GRID} x {mi.GRID} cells of 16 m, side {side}; band {spec.band_m:g} m "
                 f"(green: our zone, red: enemy, dotted: band)")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return path


def zoom_png(data, spec, path, box, side=1):
    """One close-up: field share + zone flags inside box (xmin, xmax, zmin, zmax) with the cell grid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lay = mi.unpack(data, side)
    x0, z0 = data["meta"]["origin"]
    ext = (x0, x0 + mi.SIZE_M, z0, z0 + mi.SIZE_M)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, k, t in zip(axes, (0, 2, 5), ("1 field share", "3 our zone (wholly inside)", "6 enemy vanguard")):
        ax.imshow(lay[k], origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
        poly = spec.zones[side]
        p = np.vstack([poly, poly[:1]])
        ax.plot(p[:, 0], p[:, 1], "-", color="lime", lw=1.5)
        b = _band_outline(poly, spec.band_m)
        b = np.vstack([b, b[:1]])
        ax.plot(b[:, 0], b[:, 1], ":", color="lime", lw=1.5)
        if spec.field_box is not None:
            fx0, fx1, fz0, fz1 = spec.field_box
            ax.plot([fx0, fx1, fx1, fx0, fx0], [fz0, fz0, fz1, fz1, fz0], "w--", lw=1.5)
        for g in np.arange(x0, x0 + mi.SIZE_M + 1, mi.CELL_M):
            ax.axvline(g, color="k", lw=0.3)
        for g in np.arange(z0, z0 + mi.SIZE_M + 1, mi.CELL_M):
            ax.axhline(g, color="k", lw=0.3)
        ax.set_xlim(box[0], box[1])
        ax.set_ylim(box[2], box[3])
        ax.set_title(t)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return path
