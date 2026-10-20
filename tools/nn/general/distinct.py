"""The classifier of missions by behaviour (run.py --general-distinct W; docs/en/training/general.md "Training"): what
the units of a group did during a plan (runtime.Behaviour: their order kinds' shares, the run share, the group's
displacement forward / sideways from the side's facing, its path) -> the mission kind the group had. The executor's
bonus is how well it guesses (signal.distinct_bonus): a reason for the executor to act differently under different
missions, not a payment for carrying one out (the general's return stays the battle's reward).

Its own small network and Adam, outside the executor (not in the checkpoint's actor: an old step loads and plays as
before); its weights, Adam and the decay's counter go on in the checkpoint's "train" (run.py blob).
"""
import torch
from torch import nn

from tools.nn.general import signal


class Distinct(nn.Module):
    def __init__(self, features, kinds, hidden=64):
        super().__init__()
        self.kinds = kinds
        self.net = nn.Sequential(nn.Linear(features, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, kinds))
        nn.init.zeros_(self.net[-1].weight)              # starts uniform: the bonus starts at 0
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, beh):
        """beh [..., F] -> log q(kind | behaviour) [..., K]."""
        return torch.log_softmax(self.net(beh.float()), -1)


class Learner:
    """The classifier with its Adam: bonus() at a plan's end (rollout), fit() on the update's ended plans."""

    def __init__(self, features, kinds, device, lr=1e-3, epochs=2, minibatch=1024):
        self.net = Distinct(features, kinds).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.epochs, self.minibatch = epochs, minibatch
        self.weight = 0.0                                # the bonus's weight now (run.py: signal.distinct_weight)

    @torch.no_grad()
    def bonus(self, beh, kind, ok):
        """The groups' bonus [M, G] (0 where not ok) of behaviour beh [M, G, F], their kinds [M, G] (long), ok [M, G]
        (own groups with members seen during the plan) at the current weight."""
        lq = self.net(beh).gather(-1, kind[..., None])[..., 0]
        return signal.distinct_bonus(lq, self.net.kinds, self.weight) * ok.to(lq.dtype)

    def fit(self, ready):
        """Cross-entropy steps on the ended plans' groups (ready: records with "beh", "beh_ok", "kind"). -> stats:
        accuracy and loss before the steps, the mean log q + log K (the unweighted bonus), groups."""
        if ready is None or "beh" not in ready:
            return {}
        ok = ready["beh_ok"].reshape(-1)
        x = ready["beh"].reshape(-1, ready["beh"].shape[-1])[ok]
        y = ready["kind"].reshape(-1)[ok]
        n = int(x.shape[0])
        if n == 0:
            return {"distinct_groups": 0.0}
        self.net.train()
        with torch.no_grad():
            lq = self.net(x)
            acc = float((lq.argmax(-1) == y).float().mean())
            loss0 = float(-lq.gather(-1, y[:, None]).mean())
            gain = float(signal.distinct_bonus(lq.gather(-1, y[:, None])[:, 0], self.net.kinds).mean())
        for _ in range(self.epochs):
            for mb in torch.randperm(n, device=x.device).split(self.minibatch):
                loss = -self.net(x[mb]).gather(-1, y[mb][:, None]).mean()
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                self.opt.step()
        self.net.eval()
        return {"distinct_acc": acc, "distinct_loss": loss0, "distinct_gain": gain, "distinct_groups": float(n),
                "distinct_chance": 1.0 / self.net.kinds}

    def state(self):
        return {"net": self.net.state_dict(), "opt": self.opt.state_dict()}

    def load(self, st):
        """Weights and Adam from a checkpoint's "train" (None: fresh). -> whether loaded."""
        if not st:
            return False
        self.net.load_state_dict(st["net"])
        self.opt.load_state_dict(st["opt"])
        return True
