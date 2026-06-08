# Research Boundary

Embryo currently supports a bounded diagnostic claim:

> Under a frozen deployable observation contract, `learned_fact_writer_v0`
> improves Crafter memory-evaluability and produces a large diagnostic
> clean-vs-control score under the prior collapsed-control long-run protocol.

This is a mechanism result, not a task-performance result.

## What This Means

- The actor/query stack uses deployable observations rather than privileged environment state.
- The current supported claim compares clean memory against no-memory and a
  collapsed corrupt-memory control.
- The forward protocol now exposes separate shuffled-memory, stale-memory, and
  wrong-binding controls for validation. They do not support a stronger claim
  until they are independently run and inspected.
- The stack was evaluated in real closed-loop Crafter replay.
- The score measures memory-grounded follow-through, not benchmark reward.

## What This Does Not Mean

- It does not mean the agent plays Crafter competently.
- It does not claim reward or achievement benchmark success.
- The current memory claim does not rely on PPO, reinforcement learning, or
  online fine-tuning.
- A separate experimental Crafter survival actor training lane exists. Its
  current results do not support a memory, competence, or benchmark claim.
- It does not demonstrate transfer to Minecraft, Craftax, or robotics.
- It does not allow reward, done flags, semantic maps, privileged backend
  inventory, player position, achievements, source metadata, or backend
  internals as actor/query inputs.
- A deployable inventory belief may be added as a future memory surface only if
  it is inferred from observations or agent-owned events, not read from backend
  simulator state.

Teacher diagnostics may be used for labels and evaluation, but not as deployable actor inputs.
