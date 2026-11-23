"""The fit (tmp/engine/design_merged.md sections 0, 4): one global minibatch permutation of the mixture's weighted
rows, the loss through record.loss on the level's adapter.

    passes(level, net, opt, rows, cards, w, dev, log, gen, n, mb, max_grad, pre_pass, batch_of)   n passes over the
                rows with weight (w.sum(1) > 0) in minibatches of mb, the permutation from gen (torch.randperm);
                pre_pass - a pass -1 with the losses before any update; each minibatch a Batch from batch_of(rows,
                cards, idx) (default level.batch_of) moved to dev. Returns (the plain mean loss a pass - the legacy
                imitate.fit's numbers on the same rows / generator -, {term: [mean a pass]}); the log row's "loss" is
                the pass loss weighted by each minibatch's weight sum (NEEDS_A 2: the
                plain mean is ruled by a short last minibatch), "loss_plain" next to it
    first_minibatch(level, net, rows, cards, w, dev, mb, batch_of)   the first minibatch's Batch, logits and losses
                (the step-2 equality check)
The schedule `rounds` is driven by run.py; `aggregate` is not built (run.py refuses it).
"""
import time

import torch

from net.drill import record as RC


def netbuffer_batch(rows, cards, idx):
    """A NetBuffer-keyed level's batch maker (record.from_netbuffer: the per-battle card gathered per row)."""
    return RC.from_netbuffer({**rows, "cards": cards}, idx)


def _maker(level, batch_of):
    return batch_of or getattr(level, "batch_of", None) or netbuffer_batch


def passes(level, net, opt, rows, cards, w, dev, log, gen, n, mb, max_grad, pre_pass=False, batch_of=None):
    mk = _maker(level, batch_of)
    use = (w.sum(1) > 0).nonzero()[:, 0]
    hist, terms = [], {}
    for p in range(-1 if pre_pass else 0, n):
        net.train(p >= 0)
        perm = use[torch.randperm(use.numel(), generator=gen)] if p >= 0 else use
        tot, nb, t0 = 0.0, 0, time.time()
        tw, ws = 0.0, 0.0
        acc = {}
        for i in range(0, perm.numel(), mb):
            idx = perm[i:i + mb]
            batch = mk(rows, cards, idx).to(dev)
            wi = w[idx].to(dev)
            with torch.set_grad_enabled(p >= 0):
                loss, tm = RC.loss(level, net, batch, wi)
            if p >= 0:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_([q for q in net.parameters() if q.requires_grad], float(max_grad))
                opt.step()
            tot += float(loss.detach())
            wsum = float(wi.sum())
            tw += float(loss.detach()) * wsum
            ws += wsum
            for k, v in tm.items():
                acc[k] = acc.get(k, 0.0) + (0.0 if v is None else float(v.detach()))
            nb += 1
        lw = round(tw / max(ws, 1e-9), 5)
        if p >= 0:
            hist.append(round(tot / max(nb, 1), 5))
        row = {"pass": p, "loss": lw, "loss_plain": round(tot / max(nb, 1), 5), "minibatches": nb,
               "min": round((time.time() - t0) / 60, 1)}
        for k, v in acc.items():
            terms.setdefault(k, []).append(round(v / max(nb, 1), 5))
            row[k] = terms[k][-1]
        log(row)
    net.eval()
    return hist, terms


def first_minibatch(level, net, rows, cards, w, dev, mb, batch_of=None):
    """(Batch, {the level's logits}, {term: loss}) of the first minibatch in the pre-pass order (use[:mb]),
    eval mode, no grad."""
    use = (w.sum(1) > 0).nonzero()[:, 0]
    idx = use[:mb]
    batch = _maker(level, batch_of)(rows, cards, idx).to(dev)
    net.eval()
    with torch.no_grad():
        lo = level.logits(net, batch)
        _, tm = RC.loss(level, net, batch, w[idx].to(dev))
    return batch, {k: v for k, v in lo.items() if k != "enc"}, tm
