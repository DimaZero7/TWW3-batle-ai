"""Learning by observation (docs/en/training/training.md "Learning by observation"): a recorded battle of another
player (a human against the network, the game's AI against the game's AI) -> the network's input of that player's
side, his orders read from the recording in the network's language, and how good each turned out by the network's own
critic (record.py, labels.py, advantage.py, convert.py); in training an extra imitation term on a separate batch of
these demonstrations, only the ones better than the critic expected (store.py, loss.py; tools/nn/train/run.py
--observe-dir)."""
