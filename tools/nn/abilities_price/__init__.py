"""An ability press priced by what its bonuses did (docs/en/training/ability_price.md).

Three layers:
    bonuses.py   layer 1 - one brick per bonus kind: a counter of the bonus's useful work on a trajectory (>= 0) and its
                 declared description (condition, proportional or not, buff or debuff)
    ability.py   layer 2 - an ability as data: its bricks with their numbers, duration, recharge, radius, targets (from
                 config/nn/abilities.json); the press's price = sum of rate x counter - debuffs - recharge cost
    validate.py  layer 3 - one generic validator: tests built from the bricks' descriptions, distributions over real
                 presses, agreement with the open-loop branch prices, the rates k by regression, a validity card
    collect.py   the data: simulator battles of a network, its presses (and moments it left a ready ability), each
                 replayed open loop (tools/nn/train/ability_branch.py) with the pressed branch's trajectory recorded
                 (torch; the training container)

    lab.py       the bonus lab: each kind's rate k (and its curves) measured in controlled simulator pairs that differ
                 only by the bonus, independent of the network's presses; the rates' layer-3 check (numpy)
    lab_sim.py   the lab's simulator runs: small scripted fights and whole battles, pairs on / off (torch)

Only numpy in the first three and lab.py (the host's tests); collect.py and lab_sim.py need torch.
"""
