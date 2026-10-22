"""The executor ("Исполнитель"): today's unit network (tools/nn/model/policy.py ActorV2) with one new input, the unit's
task ("Задание отряда", GeneralConfig.task_dim numbers a unit), and the general's network inside it (net.General), so
a checkpoint, the pool's snapshots and the optimizer carry both. Its chained heads are taskmask.MaskedChainHeads: the
task's masks cut a unit's choices to its mission (GeneralConfig.task_mask); without the task's numbers in the
observation they compute what chain.ChainHeads does.

The task goes through one linear layer (task_in, cfg.d wide) whose weights and bias start at zero and is added to the
unit's token where the unit has a task (obs "task_on": an own unit in a group; enemies never), right where the
commitment's input is added (ActorV2.encode, copied here with that one line more). With zero weights - or without the
"task" input - every output is the old network's to the bit (test_general_executor_torch.py).

    actor = executor.wrap(old_actor, GeneralConfig())     # the old weights, a fresh general and task_in at zero
"""
import torch
from torch import nn

from tools.nn.general.config import GeneralConfig
from tools.nn.general.net import General
from tools.nn.model import policy as model_policy

# the parameters an executor has that a unit network's checkpoint has not: they start fresh when it is loaded
FRESH = ("task_in.", "general.")
# the general's layers added after the atlas's repairs (net.General: fight_in, gtile_in): an older general's checkpoint
# loads them at zero, with a warning (once)
NEW_LAYERS = ("general.fight_in.", "general.gtile_in.")
_WARNED = []


class GeneralActor(model_policy.ActorV2):
    def __init__(self, cfg, gcfg=None):
        super().__init__(cfg)
        assert cfg.sectors, "the general's executor is a v2 unit network (ModelConfig.sectors > 0)"
        self.gcfg = gcfg or GeneralConfig()
        # the chained heads with the task's masks (taskmask.py): the same module and weights, the masks only where the
        # observation carries the task's numbers (Runtime.task with GeneralConfig.task_mask)
        from tools.nn.general.taskmask import MaskedChainHeads
        self.heads.__class__ = MaskedChainHeads
        from tools.nn.general.taskmask import parts_of
        self.heads.parts = parts_of(self.gcfg)                  # the masks' parts in force (GeneralConfig.task_mask_parts)
        # (registered after every part of the unit network: their parameters come last, run.resume keeps the order)
        self.task_in = nn.Linear(self.gcfg.task_dim, cfg.d)
        nn.init.zeros_(self.task_in.weight)
        nn.init.zeros_(self.task_in.bias)
        self.general = General(self.gcfg, cfg.sectors, cfg.grid_half)

    def encode(self, obs_t):
        """ActorV2.encode with the task added to the unit tokens (module doc)."""
        from tools.nn.model import pairs, sectors
        from tools.nn.model.encoder import attention_bias
        x = self.encoder(obs_t["tokens"], obs_t["ctx"])
        pad = (lambda v: torch.nn.functional.pad(v, (0, 0, 1, 0)))
        if obs_t.get("abil") is not None:
            x = x + pad(self.ability_input(obs_t))
        if obs_t.get("commit") is not None:
            x = x + pad(self.commit_in(obs_t["commit"].to(x.dtype)))
        if obs_t.get("task") is not None:
            on = obs_t["task_on"][..., None].to(x.dtype)
            x = x + pad(self.task_in(obs_t["task"].to(x.dtype)) * on)
        pf = pairs.features(obs_t).to(x.dtype) if self.cfg.pairs else None
        bias = attention_bias(obs_t, self.dist, self.cfg.dist_bins, None if pf is None else self.pair_bias(pf))
        g = obs_t.get("geo")
        S, k2 = self.cfg.sectors ** 2, self.cfg.fine ** 2
        cells_in = (sectors.inside(self.cfg, g) if g is not None
                    else torch.ones((x.shape[0], S, k2), dtype=torch.bool, device=x.device))
        f = sectors.features(self.cfg, obs_t, cells_in).to(x.dtype)
        buckets = self.sector_att.buckets(obs_t)
        s = None
        for i, block in enumerate(self.blocks[:self._split()]):
            if i == self.cfg.sector_at:
                x, s = self._look(x, f, buckets)
            x = block(x, bias)
        if self.cfg.sector_at == self._split():
            x, s = self._look(x, f, buckets)
        return x, (bias, s, cells_in, pf)

    def set_rules(self, gcfg):
        """Another GeneralConfig of the same shape (run.general_config: options over a checkpoint's rules - the
        deployment, the re-planning, the collection's rounds): every part that holds the config gets it."""
        keep = ("card", "reader_hidden", "council_layers", "council_heads", "group_layers", "ff", "places", "kinds",
                "target_grid", "center", "sharp", "stick", "tiles",
                "effects")
        changed = [k for k in keep if getattr(gcfg, k) != getattr(self.gcfg, k)]
        if changed:
            raise ValueError(f"set_rules: {changed} change the general's shape (its weights would not fit)")
        from tools.nn.general.taskmask import parts_of
        self.gcfg = gcfg
        self.general.cfg = gcfg
        self.heads.parts = parts_of(gcfg)
        self.general.groups.rounds = gcfg.rounds

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """As Actor's; a unit network's checkpoint (no task_in, no general) loads with those fresh."""
        missing, unexpected = nn.Module.load_state_dict(self, state_dict, strict=False, assign=assign)
        bad = [k for k in missing if not k.startswith(model_policy.FRESH_PARAMS + FRESH)] + list(unexpected)
        if strict and bad:
            raise RuntimeError(f"actor state does not match: {bad[:5]}")
        if "general.exec_kinds" in missing and "general.kinds" in state_dict:
            # a checkpoint from before the executor's own kind table (GeneralConfig.split_kinds): it starts as the
            # general's table (the executor reads the kinds it was trained on)
            with torch.no_grad():
                self.general.exec_kinds.copy_(self.general.kinds)
        new = sorted({k.split(".")[1] for k in missing if k.startswith(NEW_LAYERS)})
        if new and any(k.startswith("general.") for k in state_dict) and not _WARNED:
            _WARNED.append(1)
            print("general: a checkpoint without the layers " + ", ".join(new) + " (net.General: the fight's numbers, "
                  "the tiles under the groups) - they start fresh at zero: it plays as before until they learn",
                  flush=True)
        return missing, unexpected


def has_task(actor):
    return isinstance(actor, GeneralActor)


def wrap(actor, gcfg=None):
    """The executor of a unit network: its weights (the task input at zero, a fresh general), on its device, in its
    mode. An executor is returned as it is."""
    if isinstance(actor, GeneralActor):
        return actor
    dev = next(actor.parameters()).device
    new = GeneralActor(actor.cfg, gcfg)
    new.load_state_dict(actor.state_dict())
    new = new.to(dev)
    return new.train(actor.training)


def general_params(actor):
    """[parameters] of the general's network (their own Adam group: run.py --general-lr-mult; the task input learns
    at the executor's rate)."""
    return [p for n, p in actor.named_parameters() if n.startswith("general.")]
