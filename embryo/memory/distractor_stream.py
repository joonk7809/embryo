"""Budgeted fact memory for distractor-stream write-selection probes."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StreamFact:
    """A deployable fact plus eval-only relevance label."""

    key: int
    value: int
    cue: float
    features: tuple[int, ...]
    stream_index: int
    relevant_eval_only: bool

    def actor_view(self) -> dict[str, Any]:
        return {
            "stream_fact": {
                "key": int(self.key),
                "value": int(self.value),
                "cue": float(self.cue),
                "features": [int(value) for value in self.features],
                "stream_index": int(self.stream_index),
            }
        }


@dataclass(frozen=True)
class StreamEpisode:
    seed: int
    distractor_ratio: int
    budget: int
    p_cue: float
    cue_model: str
    cue_true_positive_rate: float
    cue_false_positive_rate: float
    feature_count: int
    feature_true_positive_rate: float
    feature_false_positive_rate: float
    relevant_count_mode: str
    facts: tuple[StreamFact, ...]
    queries: tuple[StreamFact, ...]


class BudgetedFactMemory:
    """Admission-only memory with fixed FIFO eviction."""

    def __init__(self, *, budget: int) -> None:
        if int(budget) <= 0:
            raise ValueError("budget must be positive")
        self.budget = int(budget)
        self._entries: list[StreamFact] = []
        self.admitted_count = 0

    def admit(self, fact: StreamFact) -> None:
        self.admitted_count += 1
        self._entries.append(fact)
        if len(self._entries) > self.budget:
            self._entries.pop(0)

    def recall(self, key: int) -> StreamFact | None:
        for fact in reversed(self._entries):
            if int(fact.key) == int(key):
                return fact
        return None

    def entries(self) -> tuple[StreamFact, ...]:
        return tuple(self._entries)

    def content_hash(self) -> str:
        payload = [
            {"key": fact.key, "value": fact.value, "cue": fact.cue, "stream_index": fact.stream_index}
            for fact in self._entries
        ]
        return hashlib.sha1(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]


def make_distractor_stream_episode(
    *,
    seed: int,
    relevant_count: int,
    budget: int,
    distractor_ratio: int,
    p_cue: float,
    cue_model: str = "symmetric_label_accuracy",
    cue_true_positive_rate: float | None = None,
    cue_false_positive_rate: float | None = None,
    feature_count: int = 4,
    feature_true_positive_rate: float = 0.75,
    feature_false_positive_rate: float = 0.25,
    key_space: int,
    value_space: int,
) -> StreamEpisode:
    if int(relevant_count) <= 0:
        raise ValueError("relevant_count must be positive")
    if int(budget) < int(relevant_count):
        raise ValueError("Phase-0 oracle ceiling requires budget >= relevant_count")
    if int(feature_count) <= 0:
        raise ValueError("feature_count must be positive")
    total_count = int(relevant_count) * (int(distractor_ratio) + 1)
    if int(key_space) < total_count:
        raise ValueError("key_space must cover stream length without key reuse")
    true_positive_rate, false_positive_rate = cue_rates(
        p_cue=float(p_cue),
        cue_model=cue_model,
        cue_true_positive_rate=cue_true_positive_rate,
        cue_false_positive_rate=cue_false_positive_rate,
    )
    rng = random.Random(int(seed))
    relevant_positions = set(rng.sample(range(total_count), int(relevant_count)))
    keys = rng.sample(range(int(key_space)), total_count)
    facts: list[StreamFact] = []
    for index in range(total_count):
        relevant = index in relevant_positions
        features = weak_feature_vector(
            rng,
            relevant=relevant,
            feature_count=int(feature_count),
            true_positive_rate=float(feature_true_positive_rate),
            false_positive_rate=float(feature_false_positive_rate),
        )
        facts.append(
            StreamFact(
                key=int(keys[index]),
                value=int(rng.randrange(int(value_space))),
                cue=cue_for_relevance(
                    rng,
                    relevant=relevant,
                    true_positive_rate=true_positive_rate,
                    false_positive_rate=false_positive_rate,
                ),
                features=features,
                stream_index=index,
                relevant_eval_only=bool(relevant),
            )
        )
    queries = tuple(fact for fact in facts if fact.relevant_eval_only)
    return StreamEpisode(
        seed=int(seed),
        distractor_ratio=int(distractor_ratio),
        budget=int(budget),
        p_cue=float(p_cue),
        cue_model=str(cue_model),
        cue_true_positive_rate=true_positive_rate,
        cue_false_positive_rate=false_positive_rate,
        feature_count=int(feature_count),
        feature_true_positive_rate=clamp01(feature_true_positive_rate),
        feature_false_positive_rate=clamp01(feature_false_positive_rate),
        relevant_count_mode="fixed_exact",
        facts=tuple(facts),
        queries=queries,
    )


def cue_rates(
    *,
    p_cue: float,
    cue_model: str,
    cue_true_positive_rate: float | None = None,
    cue_false_positive_rate: float | None = None,
) -> tuple[float, float]:
    """Resolve cue semantics as P(cue=1|relevant), P(cue=1|distractor)."""
    if cue_model == "symmetric_label_accuracy":
        bounded = clamp01(p_cue)
        return bounded, 1.0 - bounded
    if cue_model == "explicit_rates":
        if cue_true_positive_rate is None or cue_false_positive_rate is None:
            raise ValueError("explicit_rates requires cue_true_positive_rate and cue_false_positive_rate")
        return clamp01(cue_true_positive_rate), clamp01(cue_false_positive_rate)
    raise ValueError(f"Unknown cue_model: {cue_model}")


def cue_for_relevance(
    rng: random.Random,
    *,
    relevant: bool,
    true_positive_rate: float,
    false_positive_rate: float,
) -> float:
    rate = true_positive_rate if relevant else false_positive_rate
    return 1.0 if rng.random() < clamp01(rate) else 0.0


def weak_feature_vector(
    rng: random.Random,
    *,
    relevant: bool,
    feature_count: int,
    true_positive_rate: float,
    false_positive_rate: float,
) -> tuple[int, ...]:
    rate = true_positive_rate if relevant else false_positive_rate
    return tuple(1 if rng.random() < clamp01(rate) else 0 for _ in range(int(feature_count)))


def bayes_relevance_score(
    fact: StreamFact,
    *,
    relevant_base_rate: float,
    feature_true_positive_rate: float,
    feature_false_positive_rate: float,
) -> float:
    """Return log posterior odds up to a monotonic transform."""
    prior = clamp_probability(relevant_base_rate)
    tp = clamp_probability(feature_true_positive_rate)
    fp = clamp_probability(feature_false_positive_rate)
    score = math.log(prior / (1.0 - prior))
    for value in fact.features:
        if int(value):
            score += math.log(tp / fp)
        else:
            score += math.log((1.0 - tp) / (1.0 - fp))
    return score


def bayes_admission_indices(episode: StreamEpisode) -> frozenset[int]:
    """Select the budgeted top posterior-score facts from deployable features."""
    base_rate = len(episode.queries) / len(episode.facts)
    ranked = sorted(
        episode.facts,
        key=lambda fact: (
            -bayes_relevance_score(
                fact,
                relevant_base_rate=base_rate,
                feature_true_positive_rate=episode.feature_true_positive_rate,
                feature_false_positive_rate=episode.feature_false_positive_rate,
            ),
            stable_random_score(seed=episode.seed + 99173, stream_index=fact.key),
        ),
    )
    return frozenset(fact.stream_index for fact in ranked[: episode.budget])


def single_feature_admission_indices(episode: StreamEpisode, *, feature_index: int) -> frozenset[int]:
    """Select budgeted facts ranked by one observable feature only."""
    ranked = sorted(
        episode.facts,
        key=lambda fact: (
            -int(fact.features[int(feature_index)]),
            stable_random_score(seed=episode.seed + 41413 + int(feature_index), stream_index=fact.key),
        ),
    )
    return frozenset(fact.stream_index for fact in ranked[: episode.budget])


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def clamp_probability(value: float) -> float:
    return max(1e-9, min(1.0 - 1e-9, float(value)))


def random_admission_indices(*, seed: int, stream_length: int, budget: int) -> frozenset[int]:
    """Return exactly budget random stream indices with stable tie-breaking."""
    count = min(int(stream_length), int(budget))
    ranked = sorted(range(int(stream_length)), key=lambda index: (stable_random_score(seed=seed, stream_index=index), index))
    return frozenset(ranked[:count])


def stable_random_score(*, seed: int, stream_index: int) -> float:
    digest = hashlib.sha1(f"distractor-stream-random:{int(seed)}:{int(stream_index)}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(0xFFFFFFFFFFFF)
