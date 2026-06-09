# Embryo Memory

This repository is a reproduction subset for controlled explicit-memory probes
in sequential agents.

## Sized Claim

An explicit decoupled memory store can be written, learned, retrieved, and
injected into a policy decision to improve controlled-task recall accuracy and
retrieval efficiency on tasks where correct stored content is required.

## Evidence Ladder

| Rung | Probe | Result shape |
| --- | --- | --- |
| 1 | POPGym Autoencode hand-coded recall | Explicit ordered memory stays at 1.0 success after the recurrent baseline falls toward chance. |
| 2 | POPGym Autoencode learned associative read | A small learned reader recovers the hand-coded clean-memory ceiling and preserves the corruption signature. |
| 3 | POPGym CountRecall action-level injection | Clean recalled counts reach oracle accuracy; shuffled, wrong-binding, and stale controls fail. |
| 4 | Gridworld kitchen goal-channel injection | Recalled drawer locations drive a frozen goal-conditioned navigator; retrieval cost advantage grows with drawer count. |

See `docs/evidence_consolidation.md` and `docs/evidence_consolidation.pdf` for
the consolidated artifact.

## Non-Claims

This subset does not claim learned write selection, full episodic memory
management, cross-episode memory, external-backbone portability, reinforcement
learning reward improvement, Crafter competence, or broad embodied-agent
competence.

## Install

The CountRecall source receipt is tied to `popgym==1.0.6`. Keep that version
pinned unless you re-audit the source deviation and regenerate the receipt.

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e ".[all]"
```

## Reproduce

Run tests:

```sh
python3 -m compileall -q embryo scripts tests
python3 -m unittest discover -s tests
```

Run the probe suite:

```sh
python3 scripts/train_popgym_autoencode.py \
  --config configs/popgym_autoencode_recurrent.yaml \
  --out runs/popgym_autoencode_recurrent

python3 scripts/probe_popgym_count_recall_injection.py \
  --config configs/popgym_count_recall_injection.yaml \
  --out runs/popgym_count_recall_injection

python3 scripts/probe_gridworld_injection.py \
  --config configs/gridworld_kitchen_injection.yaml \
  --out runs/gridworld_kitchen_injection

python3 scripts/build_evidence_consolidation_figures.py
python3 scripts/build_evidence_consolidation_pdf.py
```

## CountRecall Deviation

The CountRecall probe is reported as corrected CountRecall, not untouched stock
CountRecall. In `popgym==1.0.6`, `CountRecallHard` can reward a target count
equal to `max_card_count` while declaring `gym.spaces.Discrete(max_card_count)`,
which excludes that value. The local correction expands the declared action
space consistently for all arms.

## License

Code is licensed under Apache-2.0. Documentation, figures, and the PDF artifact
are licensed under CC-BY-4.0; see `LICENSE-DOCS`.
