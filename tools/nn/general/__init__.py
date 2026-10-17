"""The general (S28): the battle plan network's inputs. Design: design/general/index.html.

    map_input.py   the map's layers (field / outside shares, deployment flags) on a 128 x 128 grid of 16 m cells
    map_batch.py   the same layers for many battles on the fly (torch): zones per battle (drills)
    geometry.py    exact cell-against-shape sums: share of a cell inside a shape, cell wholly inside / far from it
    render.py      check pictures of the layers (png)
    map_cnn.py     the map analysis (CNN): layers -> 1024 regions x 32 numbers; summary cache per map
    map_probe.py   checks of the map analysis now and then: dead features, crowding, twins, pictures
"""
