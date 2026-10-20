"""The general's army analysis, "Анализ армий" (design/general/index.html): ONE small dense network of 2 layers that
reads every occupied slot of the army input (army_input.py) with the same weights - own and enemy alike - and gives
each its unit card, "Карточки отрядов": [B, 80, D].

    layer 1   h = ReLU(W1 numbers + b1 + E_trajectory[t] + E_caste[c] + E_size[s])     numbers 45 -> H
    layer 2   card = W2 h + b2 (no activation: the card)                                 H -> D
    effects   card += the unit's effect cards ("Анализ эффектов", not designed yet: `effects`, default 0)

A label is read as a learned vector of the first layer (E_*: nn.Embedding H wide) - the same as its one-hot flags
times their weights. The trajectory's index 0 ("no missile weapon") is a fixed zero vector (padding_idx). H = D = 256:
45 x 256 + 256 + (4 + 12 + 4) x 256 + 256 x 256 + 256 = 82 688 weights (the design's estimate: 62 x 256 + 256 x 256).

Empty slots never pass through the network (they are taken out before it, `empty`): their card is 0 and they cannot
touch the others' cards. The slots' order means nothing: the network works per slot, a permutation of the slots
permutes the cards. Own and enemy differ only by the side flag's column of W1 (army_input.SIDE).

A new variant of a label (a new caste, size or trajectory appended to army_input's lists): add_variant() appends a
vector of zeros to its table. The old variants' rows are untouched, so every old card stays the same to the bit; a
unit of the new variant starts with no label contribution and training moves its vector.

    python -m tools.nn.general.army_reader --bench       weights, time: 1024 battles of 80 slots
"""
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.nn.general import army_input as ai


class ArmyReader(nn.Module):
    """Army input (numbers [B, 80, N], labels [B, 80, 3], empty [B, 80]) -> cards [B, 80, card] (module doc).
    hidden: the first layer's width; card: the card's size (both 256; the design: to be settled by measurement)."""

    def __init__(self, hidden=256, card=256, numbers=ai.SIZE, variants=None):
        super().__init__()
        variants = variants or {k: len(v) for k, v in ai.VARIANTS.items()}
        self.hidden, self.card, self.numbers = int(hidden), int(card), int(numbers)
        self.layer1 = nn.Linear(self.numbers, self.hidden)
        self.labels = nn.ModuleDict({k: self._table(variants[k], k) for k in ai.LABELS})
        self.layer2 = nn.Linear(self.hidden, self.card)
        # He init over the first layer's whole input (numbers + one flag a label's variant), so a label's vector is
        # on the scale of a number's column; the second layer keeps the scale (test_numbers_keep_their_scale)
        fan_in = self.numbers + sum(variants.values())
        std = (2.0 / fan_in) ** 0.5
        nn.init.normal_(self.layer1.weight, 0.0, std)
        nn.init.zeros_(self.layer1.bias)
        for k, e in self.labels.items():
            nn.init.normal_(e.weight, 0.0, std)
            if e.padding_idx is not None:
                with torch.no_grad():
                    e.weight[e.padding_idx].zero_()
        nn.init.normal_(self.layer2.weight, 0.0, (1.0 / self.hidden) ** 0.5)
        nn.init.zeros_(self.layer2.bias)

    def _table(self, n, kind=None):
        return nn.Embedding(n, self.hidden, padding_idx=0 if kind == "trajectory" else None)

    def variants(self):
        return {k: e.num_embeddings for k, e in self.labels.items()}

    def add_variant(self, kind, n=1):
        """n new variants of label `kind` (appended after the old ones) with zero vectors: old cards stay to the bit."""
        old = self.labels[kind]
        new = nn.Embedding(old.num_embeddings + n, self.hidden, padding_idx=old.padding_idx).to(
            device=old.weight.device, dtype=old.weight.dtype)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:old.num_embeddings] = old.weight
        self.labels[kind] = new
        return self

    # --- forward ------------------------------------------------------------------------------------------------
    def first(self, numbers, labels):
        """Layer 1 before its ReLU, rows of slots: numbers [M, N], labels [M, 3] -> [M, H]."""
        h = self.layer1(numbers)
        for i, k in enumerate(ai.LABELS):
            h = h + self.labels[k](labels[:, i])
        return h

    def read(self, numbers, labels):
        """Cards of occupied slots as rows: [M, N], [M, 3] -> [M, D]."""
        return self.layer2(F.relu(self.first(numbers, labels)))

    def forward(self, numbers, labels, empty, effects=None):
        """numbers [B, 80, N] float, labels [B, 80, 3] long, empty [B, 80] bool -> cards [B, 80, D]; effects [B, 80, D]
        (the effect cards, "Анализ эффектов"; None: 0) are added to the occupied slots' cards. Empty slots: 0."""
        assert numbers.shape[-1] == self.numbers, f"{numbers.shape[-1]} numbers a slot, the network reads {self.numbers}"
        lead = numbers.shape[:-1]
        full = ~empty.reshape(-1)
        rows = self.read(numbers.reshape(-1, self.numbers)[full], labels.reshape(-1, labels.shape[-1])[full])
        if effects is not None:
            rows = rows + effects.reshape(-1, self.card)[full]
        out = rows.new_zeros(full.shape[0], self.card)
        out[full] = rows
        return out.reshape(*lead, self.card)


def weights(model):
    return sum(p.numel() for p in model.parameters())


def bench(device=None, hidden=256, card=256, battles=1024):
    from tools.nn.model import passport as pp
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(0)
    model = ArmyReader(hidden, card).to(dev)
    print(f"hidden {hidden}, card {card}: weights {weights(model)}; device {dev}")
    keys = list(pp.load())
    items = [ai.build({1: [keys[(i + j) % len(keys)] for j in range(20)],
                       2: [keys[(3 * i + j) % len(keys)] for j in range(26)]}, 1 + i % 2) for i in range(64)]
    a = ai.stack([items[i % len(items)] for i in range(battles)])
    x = ai.to_tensors(a, dev)
    for grad in (False, True):
        def run():
            if grad:
                model.zero_grad(set_to_none=True)
                model(*x).square().mean().backward()
            else:
                with torch.no_grad():
                    model(*x)
        run()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(5):
            run()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        print(f"{battles} battles x 80 slots ({int((~x[2]).sum())} occupied), "
              f"{'forward + backward' if grad else 'no grad'}: {(time.perf_counter() - t) / 5 * 1e3:.1f} ms")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--card", type=int, default=256)
    ap.add_argument("--battles", type=int, default=1024)
    a = ap.parse_args(argv)
    if a.bench:
        bench(a.device, a.hidden, a.card, a.battles)


if __name__ == "__main__":
    main()
