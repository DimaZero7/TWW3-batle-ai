"""The game AI's student (docs/en/training/student.md): a v2 network (tools/nn/model, the same build as the learner)
taught by imitation on the game AI's recorded battles - every second of every unit that takes orders: the AI's new
order (read from the recording, tools/nn/observe/labels.py) or "keep" (it gave none). dataset.py: the battles, the
split by start, the labels and the commitment replayed, the batches; objective.py: the loss and the accuracies;
train.py: the run; eval.py: the accuracies on held-out starts against simple baselines, and the student's battles in
the simulator. Its checkpoint is an ordinary one: it goes into the self-play pool as it is (run.py --pool-extra)."""
