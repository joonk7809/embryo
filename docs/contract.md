# Deployable Contract

Embryo uses an explicit observation contract to separate deployable inputs from diagnostic labels.

Allowed actor and query inputs include:

- raw RGB observations
- bounded RGB deltas
- previous actions
- action-result events inferred from deployable visual change
- sparse visual facts
- query content derived from deployable facts and events
- cache age and freshness state
- router state needed for freshness-aware fallback

Forbidden actor/query/router inputs include:

- rewards
- done or discount flags
- semantic maps
- privileged backend inventory
- player position
- achievements
- source metadata
- backend internals

Teacher diagnostics can be used for labels and evaluation, but never as deployable actor inputs.
Deployable inventory belief may be added as a memory surface only when it is
inferred from observations or agent-owned events rather than backend state.
