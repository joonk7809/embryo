"""Memory-grounded diagnostic scoring.

The score is intentionally not a benchmark reward. It credits progress that
follows resource-critical clean-memory decisions, matches simple exploration
budgets, and penalizes progress from corrupted/nonmemory behavior.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


DEFAULT_CONTROL_ARMS = ("wrong_binding", "shuffled", "stale", "no_memory")


@dataclass(frozen=True)
class ScoreConfig:
    clean_arm: str = "clean"
    control_arms: tuple[str, ...] = DEFAULT_CONTROL_ARMS
    followthrough_window: int = 8
    corrupted_penalty: float = 0.5
    nonmemory_penalty: float = 0.5
    recovery_normalization: bool = True
    exploration_tolerance: int = 1


def compute_memory_grounded_scores(
    ticks: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]] = (),
    *,
    config: ScoreConfig | None = None,
) -> dict[str, Any]:
    """Compute memory-grounded scores from clean trace rows.

    Required row fields are deliberately small: `arm`, `seed`, `episode_index`
    or `episode_id`, `tick`, action, `resource_memory_critical`, and
    `diagnostic_progress_delta_teacher_only`.
    """
    cfg = config or ScoreConfig()
    arms = sorted({str(row.get("arm")) for row in ticks if row.get("arm") is not None})
    if cfg.clean_arm not in arms:
        return empty_score_result(cfg, reason="clean_arm_missing")

    by_key = index_ticks(ticks)
    by_episode = group_episode_rows(ticks)
    future_gain = future_gain_by_row(by_episode, window=cfg.followthrough_window)
    anchors = clean_divergence_anchors(ticks, cfg=cfg, by_key=by_key)
    raw_progress = raw_progress_by_arm(ticks, episodes)
    rows: list[dict[str, Any]] = []
    clean_followthrough = followthrough_for_arm(cfg.clean_arm, anchors, by_key, future_gain)

    for arm in arms:
        arm_ticks = [row for row in ticks if str(row.get("arm")) == arm]
        matched = matched_anchors_for_arm(arm, anchors, by_key, cfg=cfg)
        resource_followthrough = followthrough_for_arm(arm, anchors, by_key, future_gain)
        matched_followthrough = followthrough_for_arm(arm, matched, by_key, future_gain)
        nonmemory_progress = sum(
            max(0.0, numeric(row.get("diagnostic_progress_delta_teacher_only")))
            for row in arm_ticks
            if not bool(row.get("resource_memory_critical") or nested_flag(row, "resource_memory_critical"))
        )
        corrupted_advantage = 0.0
        if arm != cfg.clean_arm:
            divergent_gain = followthrough_for_arm(arm, matched, by_key, future_gain)
            corrupted_advantage = divergent_gain + max(0.0, resource_followthrough - clean_followthrough)
        fallback_rate = mean_bool(row.get("fallback_triggered") or row.get("fallback_used") for row in arm_ticks)
        normalized = matched_followthrough / (1.0 + fallback_rate) if cfg.recovery_normalization else matched_followthrough
        score = normalized - cfg.nonmemory_penalty * nonmemory_progress - cfg.corrupted_penalty * corrupted_advantage
        rows.append(
            {
                "arm": arm,
                "raw_progress": round(raw_progress.get(arm, 0.0), 4),
                "resource_followthrough_progress": round(resource_followthrough, 4),
                "resource_followthrough_progress_matched": round(matched_followthrough, 4),
                "nonmemory_exploration_progress": round(nonmemory_progress, 4),
                "corrupted_binding_advantage": round(corrupted_advantage, 4),
                "fallback_rate": round(fallback_rate, 4),
                "memory_grounded_score": round(score, 4),
                "anchor_count": len(anchors),
                "matched_anchor_count": len(matched),
                "unmatched_anchor_count": max(0, len(anchors) - len(matched)),
            }
        )

    scores = {row["arm"]: row["memory_grounded_score"] for row in rows}
    raw = {row["arm"]: row["raw_progress"] for row in rows}
    gaps = clean_minus_controls(scores, cfg.clean_arm, cfg.control_arms)
    raw_gaps = clean_minus_controls(raw, cfg.clean_arm, cfg.control_arms)
    return {
        "config": config_to_dict(cfg),
        "arms": arms,
        "score_rows": rows,
        "memory_grounded_score_by_arm": scores,
        "raw_progress_by_arm": raw,
        "memory_grounded_gaps_clean_minus_controls": gaps,
        "raw_gaps_clean_minus_controls": raw_gaps,
        "best_arm": max(scores, key=scores.get) if scores else None,
        "anchor_count": len(anchors),
        "matched_exploration": matched_exploration_summary(rows, cfg),
    }


def compute_window_sensitivity(
    ticks: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]] = (),
    *,
    windows: Iterable[int] = (4, 8, 16),
    config: ScoreConfig | None = None,
) -> dict[str, Any]:
    cfg = config or ScoreConfig()
    by_window: dict[str, Any] = {}
    stable: list[int] = []
    for window in windows:
        result = compute_memory_grounded_scores(ticks, episodes, config=replace_config(cfg, followthrough_window=int(window)))
        gaps = result["memory_grounded_gaps_clean_minus_controls"]
        separates = all((gaps.get(arm) or -1.0) > 0.0 for arm in cfg.control_arms)
        if separates:
            stable.append(int(window))
        by_window[str(window)] = {
            "scores": result["memory_grounded_score_by_arm"],
            "gaps": gaps,
            "separates_clean": separates,
        }
    return {
        "windows": [int(window) for window in windows],
        "stable_windows": stable,
        "stable_for_at_least_two_windows": len(stable) >= 2,
        "by_window": by_window,
    }


def simple_memory_grounded_gap(clean: float, control: float) -> float:
    """Return clean-minus-control score gap."""
    return float(clean) - float(control)


def index_ticks(ticks: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int, str, int], Mapping[str, Any]]:
    return {row_key(row): row for row in ticks}


def group_episode_rows(ticks: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int, str], list[Mapping[str, Any]]]:
    grouped: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in ticks:
        arm, seed, episode, _ = row_key(row)
        grouped[(arm, seed, episode)].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row.get("tick", 0)))
    return grouped


def clean_divergence_anchors(
    ticks: Sequence[Mapping[str, Any]],
    *,
    cfg: ScoreConfig,
    by_key: Mapping[tuple[str, int, str, int], Mapping[str, Any]],
) -> list[tuple[int, str, int]]:
    anchors: list[tuple[int, str, int]] = []
    for row in ticks:
        arm, seed, episode, tick = row_key(row)
        if arm != cfg.clean_arm:
            continue
        if not bool(row.get("resource_memory_critical") or nested_flag(row, "resource_memory_critical")):
            continue
        clean_action = action_name(row)
        if any(
            action_name(by_key.get((control, seed, episode, tick), {})) != clean_action
            for control in cfg.control_arms
            if (control, seed, episode, tick) in by_key
        ):
            anchors.append((seed, episode, tick))
    return anchors


def matched_anchors_for_arm(
    arm: str,
    anchors: Sequence[tuple[int, str, int]],
    by_key: Mapping[tuple[str, int, str, int], Mapping[str, Any]],
    *,
    cfg: ScoreConfig,
) -> list[tuple[int, str, int]]:
    matched: list[tuple[int, str, int]] = []
    for seed, episode, tick in anchors:
        clean = by_key.get((cfg.clean_arm, seed, episode, tick))
        other = by_key.get((arm, seed, episode, tick))
        if not clean or not other:
            continue
        if abs(exploration_bin(clean) - exploration_bin(other)) <= cfg.exploration_tolerance:
            matched.append((seed, episode, tick))
    return matched


def future_gain_by_row(
    by_episode: Mapping[tuple[str, int, str], Sequence[Mapping[str, Any]]],
    *,
    window: int,
) -> dict[tuple[str, int, str, int], float]:
    gains: dict[tuple[str, int, str, int], float] = {}
    for rows in by_episode.values():
        for index, row in enumerate(rows):
            gain = 0.0
            start = int(row.get("tick", 0))
            for future in rows[index : index + window + 1]:
                if int(future.get("tick", 0)) - start > window:
                    break
                gain += max(0.0, numeric(future.get("diagnostic_progress_delta_teacher_only")))
            gains[row_key(row)] = gain
    return gains


def followthrough_for_arm(
    arm: str,
    anchors: Sequence[tuple[int, str, int]],
    by_key: Mapping[tuple[str, int, str, int], Mapping[str, Any]],
    future_gain: Mapping[tuple[str, int, str, int], float],
) -> float:
    total = 0.0
    for seed, episode, tick in anchors:
        key = (arm, seed, episode, tick)
        if key in by_key:
            total += future_gain.get(key, 0.0)
    return total


def raw_progress_by_arm(ticks: Sequence[Mapping[str, Any]], episodes: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if episodes:
        totals: dict[str, list[float]] = defaultdict(list)
        for episode in episodes:
            arm = str(episode.get("arm"))
            totals[arm].append(numeric(episode.get("diagnostic_progress_score_teacher_only")))
        return {arm: sum(values) / len(values) for arm, values in totals.items() if values}
    totals = defaultdict(float)
    for row in ticks:
        totals[str(row.get("arm"))] += numeric(row.get("diagnostic_progress_delta_teacher_only"))
    return dict(totals)


def clean_minus_controls(scores: Mapping[str, float], clean_arm: str, controls: Sequence[str]) -> dict[str, float | None]:
    clean = scores.get(clean_arm)
    return {arm: None if clean is None or scores.get(arm) is None else round(clean - float(scores[arm]), 4) for arm in controls}


def matched_exploration_summary(rows: Sequence[Mapping[str, Any]], cfg: ScoreConfig) -> dict[str, Any]:
    total = sum(int(row.get("anchor_count", 0)) for row in rows if row.get("arm") != cfg.clean_arm)
    unmatched = sum(int(row.get("unmatched_anchor_count", 0)) for row in rows if row.get("arm") != cfg.clean_arm)
    return {
        "total_anchor_pairs": total,
        "unmatched_anchor_pairs": unmatched,
        "dropped_unmatched_sample_rate": round(unmatched / total, 4) if total else 0.0,
    }


def row_key(row: Mapping[str, Any]) -> tuple[str, int, str, int]:
    arm = str(row.get("arm"))
    seed = int(row.get("seed", 0))
    episode = str(row.get("episode_id", row.get("episode_index", "0")))
    tick = int(row.get("tick", 0))
    return arm, seed, episode, tick


def action_name(row: Mapping[str, Any]) -> str:
    action = row.get("action", row.get("selected_action"))
    if isinstance(action, Mapping):
        return str(action.get("action_name", action.get("name", "")))
    return str(action)


def exploration_bin(row: Mapping[str, Any]) -> int:
    if row.get("exploration_bin") is not None:
        return int(row["exploration_bin"])
    entropy = numeric(row.get("action_entropy_proxy"))
    return max(0, min(4, int(entropy * 5)))


def nested_flag(row: Mapping[str, Any], name: str) -> bool:
    flags = row.get("critical_flags")
    return bool(flags.get(name)) if isinstance(flags, Mapping) else False


def numeric(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def mean_bool(values: Iterable[Any]) -> float:
    vals = [bool(value) for value in values]
    return sum(vals) / len(vals) if vals else 0.0


def empty_score_result(cfg: ScoreConfig, *, reason: str) -> dict[str, Any]:
    return {
        "config": config_to_dict(cfg),
        "error": reason,
        "arms": [],
        "score_rows": [],
        "memory_grounded_score_by_arm": {},
        "raw_progress_by_arm": {},
        "memory_grounded_gaps_clean_minus_controls": {},
        "raw_gaps_clean_minus_controls": {},
        "best_arm": None,
        "anchor_count": 0,
        "matched_exploration": {"total_anchor_pairs": 0, "unmatched_anchor_pairs": 0, "dropped_unmatched_sample_rate": 0.0},
    }


def config_to_dict(cfg: ScoreConfig) -> dict[str, Any]:
    return {
        "clean_arm": cfg.clean_arm,
        "control_arms": list(cfg.control_arms),
        "followthrough_window": cfg.followthrough_window,
        "corrupted_penalty": cfg.corrupted_penalty,
        "nonmemory_penalty": cfg.nonmemory_penalty,
        "recovery_normalization": cfg.recovery_normalization,
        "exploration_tolerance": cfg.exploration_tolerance,
    }


def replace_config(cfg: ScoreConfig, **updates: Any) -> ScoreConfig:
    values = config_to_dict(cfg)
    values.update(updates)
    values["control_arms"] = tuple(values["control_arms"])
    return ScoreConfig(**values)
