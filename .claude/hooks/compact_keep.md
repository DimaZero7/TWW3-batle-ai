What this summary must keep (project TWW3 battle AI).

The reader is the same agent continuing the same work with no other memory of this conversation. The project rules
(CLAUDE.md, COURSE.md, METHOD.md, game/manual.md) and the memory index reload by themselves: do not restate them.
Keep what exists only in this conversation, in this order of importance:

1. Role and authority. Is this session the orchestrator, its replica, a fork or a worker; who it reports to; ids and
   titles of linked sessions and agents; what authority was handed over, to whom and when. For a worker: its brief
   verbatim, its acceptance lines, what is done and what is not.
2. The user's words. Every user message verbatim, in Russian, typos as typed (they are short). Standing permissions
   and prohibitions with their date and scope: usage ceiling, containers, game and desktop, Docker, what is never
   committed, what is never deleted.
3. Decisions in force. For each: the decision line written BEFORE the run, the raw number that came back, the verdict,
   and the file and section where it is recorded (review*.md, net/drills.json, memory). A number is kept only with how
   it was computed: paired or unpaired, which set, which seeds, which commit, simulator or game.
4. Open work, exactly. What is running right now (agent ids and names, container names, background waits and their
   done-files, locks); what is queued and in what order; what waits for what (GPU, a night without the user, the usage
   limit's reset with its date and hour).
5. Corrections. Every mistake the user or a critic caught, with the corrected value, so the wrong value is never
   repeated as fact.
6. Places. Exact paths of files written or changed, commits (hash and one line), raw result folders under tmp/,
   weights with their hashes, the commands that reproduce a result.
7. The machine as last observed, with the time of the observation: usage percent (all models, Fable), GPU present or
   not, Docker up or not, the user at the computer or away.

Drop: tool output already folded into a decision, file contents that are on disk, abandoned options, intermediate
reasoning. Mark anything not verified in this conversation as "unverified". Give times as absolute dates and clock
times, never "today" or "an hour ago". End with the single next step, quoted from the most recent user request.
