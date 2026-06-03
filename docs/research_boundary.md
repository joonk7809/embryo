# Research Boundary

Embryo currently supports a bounded diagnostic claim:

> Under a frozen deployable observation contract, a learned RGB-derived fact writer, learned router/freshness scheduler, and learned behavior-cloned policy preserved memory-grounded separation from corrupted/no-memory controls in real Crafter diagnostic closed-loop replay.

This is a mechanism result, not a task-performance result.

## What This Means

- The actor/query stack uses deployable observations rather than privileged environment state.
- Corrupted memory controls such as no-memory, shuffled memory, stale memory, and wrong binding are separated by the diagnostic score.
- The stack was evaluated in real closed-loop Crafter replay.
- The score measures memory-grounded follow-through, not benchmark reward.

## What This Does Not Mean

- It does not mean the agent plays Crafter competently.
- It does not claim reward or achievement benchmark success.
- It does not include PPO or reinforcement learning.
- It does not include online fine-tuning.
- It does not demonstrate transfer to Minecraft, Craftax, or robotics.
- It does not allow reward, done flags, semantic maps, privileged backend
  inventory, player position, achievements, source metadata, or backend
  internals as actor/query inputs.
- A deployable inventory belief may be added as a future memory surface only if
  it is inferred from observations or agent-owned events, not read from backend
  simulator state.

Teacher diagnostics may be used for labels and evaluation, but not as deployable actor inputs.
