"""Plain statistics of the rank check (no torch): Spearman's rank correlation with ties (average ranks)."""
import math


def ranks(v):
    """Average ranks (0-based) of the values v; ties share their mean rank."""
    o = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    k = 0
    while k < len(o):
        j = k
        while j + 1 < len(o) and v[o[j + 1]] == v[o[k]]:
            j += 1
        for t in range(k, j + 1):
            r[o[t]] = (k + j) / 2
        k = j + 1
    return r


def spearman(a, b):
    """Spearman's rho of two equal lists (NaN pairs dropped; NaN when fewer than 3 or no spread)."""
    pairs = [(x, y) for x, y in zip(a, b) if not (math.isnan(x) or math.isnan(y))]
    if len(pairs) < 3:
        return float("nan")
    ra, rb = ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])
    n = len(pairs)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = math.sqrt(sum((x - ma) ** 2 for x in ra))
    vb = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return cov / (va * vb) if va and vb else float("nan")
