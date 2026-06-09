#!/usr/bin/env python
"""Build figures for the evidence consolidation document."""

from __future__ import annotations

import argparse
import json
import math
from html import escape
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs" / "assets" / "evidence_consolidation"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics = collect_metrics()
    write_autoencode_gap_curve(metrics, out / "autoencode_gap_curve.svg")
    write_autoencode_learned_read(metrics, out / "autoencode_learned_read.svg")
    write_count_recall_accuracy(metrics, out / "count_recall_accuracy.svg")
    write_gridworld_drawer_sweep(metrics, out / "gridworld_drawer_sweep.svg")
    (out / "evidence_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "figures": sorted(path.name for path in out.glob("*.svg"))}, sort_keys=True))
    return 0


def collect_metrics() -> dict[str, Any]:
    auto_hand = read_json(ROOT / "runs" / "popgym_autoencode_recurrent" / "memory_curve.json")
    auto_learned = read_json(ROOT / "runs" / "popgym_autoencode_recurrent" / "learned_memory_curve.json")
    count = read_json(ROOT / "runs" / "popgym_count_recall_injection" / "popgym_count_recall_summary.json")
    grid = read_json(ROOT / "runs" / "gridworld_kitchen_injection" / "gridworld_injection_summary.json")
    count_episodes = read_jsonl(ROOT / "runs" / "popgym_count_recall_injection" / "popgym_count_recall_episodes.jsonl")
    grid_episodes = read_jsonl(ROOT / "runs" / "gridworld_kitchen_injection" / "gridworld_injection_episodes.jsonl")
    return {
        "autoencode": {
            "hand_coded_source": "runs/popgym_autoencode_recurrent/memory_curve.json",
            "learned_source": "runs/popgym_autoencode_recurrent/learned_memory_curve.json",
            "hand_coded_status": auto_hand["status"],
            "learned_status": auto_learned["status"],
            "seed_count": auto_hand["seed_count"],
            "gaps": [int(gap) for gap in auto_hand["gaps"]],
            "hand_coded_series": {
                "off": gap_series(auto_hand, "autoencode_memory_off"),
                "clean": gap_series(auto_hand, "autoencode_memory_clean"),
                "content_corrupt": gap_series(auto_hand, "autoencode_memory_content_corrupt"),
                "order_corrupt": gap_series(auto_hand, "autoencode_memory_order_corrupt"),
                "shuffled": gap_series(auto_hand, "autoencode_memory_shuffled"),
            },
            "learned_series": {
                "off": gap_series(auto_learned, "autoencode_memory_off"),
                "hand_coded_clean": gap_series(auto_learned, "autoencode_hand_coded_memory_clean"),
                "learned_clean": gap_series(auto_learned, "autoencode_learned_memory_clean"),
                "content_corrupt": gap_series(auto_learned, "autoencode_learned_memory_content_corrupt"),
                "order_corrupt": gap_series(auto_learned, "autoencode_learned_memory_order_corrupt"),
                "shuffled": gap_series(auto_learned, "autoencode_learned_memory_shuffled"),
            },
            "hand_coded_summary_by_arm": auto_hand["summary_by_arm"],
            "learned_summary_by_arm": auto_learned["summary_by_arm"],
        },
        "count_recall": {
            "source": "runs/popgym_count_recall_injection/popgym_count_recall_summary.json",
            "decision": count["decision"],
            "runtime": count["runtime"],
            "source_receipt": count_recall_source_receipt(),
            "arms": {
                "no_memory": count["metrics_by_arm"]["count_recall_no_memory"]["query_accuracy"],
                "clean": count["metrics_by_arm"]["count_recall_memory_clean"]["query_accuracy"],
                "shuffled": count["metrics_by_arm"]["count_recall_memory_shuffled"]["query_accuracy"],
                "wrong_binding": count["metrics_by_arm"]["count_recall_memory_wrong_binding"]["query_accuracy"],
                "stale": count["metrics_by_arm"]["count_recall_memory_stale"]["query_accuracy"],
                "oracle": count["metrics_by_arm"]["count_recall_oracle_eval_only"]["query_accuracy"],
            },
            "episode_ci": episode_ci_by_arm(count_episodes, value_key="query_accuracy"),
            "pairwise": count["pairwise"],
        },
        "gridworld": {
            "source": "runs/gridworld_kitchen_injection/gridworld_injection_summary.json",
            "decision": grid["decision"],
            "runtime": grid["runtime"],
            "arms": grid["metrics_by_arm"],
            "drawers_opened_ci": episode_ci_by_arm(grid_episodes, value_key="drawers_opened_to_success", success_only=True),
            "drawers_opened_ci_by_count": gridworld_ci_by_count(grid_episodes),
            "pairwise": grid["pairwise"],
            "curve": gridworld_curve(grid),
        },
    }


def gap_series(summary: dict[str, Any], arm: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gap in sorted((int(key) for key in summary["summary_by_gap"]), key=int):
        row = summary["summary_by_gap"][str(gap)][arm]
        rows.append({"gap": gap, "success_rate": float(row["success_rate"])})
    return rows


def count_recall_source_receipt() -> dict[str, Any]:
    try:
        import popgym.envs.count_recall as module
        from popgym.envs.count_recall import CountRecallHard
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    source = Path(module.__file__).read_text(encoding="utf-8").splitlines()
    snippets = {}
    for idx, line in enumerate(source, 1):
        stripped = line.strip()
        if stripped.startswith("self.max_card_count ="):
            snippets["max_card_count_line"] = {"line": idx, "text": stripped}
        elif stripped.startswith("self.action_space ="):
            snippets["action_space_line"] = {"line": idx, "text": stripped}
        elif stripped.startswith("prev_count ="):
            snippets["target_count_line"] = {"line": idx, "text": stripped}
        elif stripped.startswith("reward = 1 if action == prev_count else -1"):
            snippets["reward_line"] = {"line": idx, "text": stripped}
    env = CountRecallHard()
    reproduction = {
        "task": "CountRecallHard",
        "action_space": str(env.action_space),
        "action_space_n": int(env.action_space.n),
        "max_card_count": int(env.max_card_count),
    }
    for seed in range(12000, 12100):
        env = CountRecallHard()
        obs, _ = env.reset(seed=seed)
        for tick in range(500):
            query = int(obs[1])
            target = int(env.counts[query])
            if target == env.action_space.n:
                _, reward, terminated, truncated, _ = env.step(target)
                reproduction.update(
                    {
                        "seed": seed,
                        "tick": tick,
                        "query": query,
                        "target_count": target,
                        "action_space_contains_target": bool(env.action_space.contains(target)),
                        "reward_for_target": float(reward),
                        "reward_positive": float(reward) > 0.0,
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                    }
                )
                return {"available": True, "module": "popgym.envs.count_recall", "snippets": snippets, "reproduction": reproduction}
            obs, _, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                break
    reproduction["reachable_out_of_space_target_found"] = False
    return {"available": True, "module": "popgym.envs.count_recall", "snippets": snippets, "reproduction": reproduction}


def episode_ci_by_arm(rows: list[dict[str, Any]], *, value_key: str, success_only: bool = False) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if success_only and not bool(row.get("retrieval_success", False)):
            continue
        value = row.get(value_key)
        if value is not None:
            grouped[str(row["arm"])].append(float(value))
    return {arm: summary_stats(values) for arm, values in sorted(grouped.items())}


def gridworld_ci_by_count(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if not bool(row.get("retrieval_success", False)):
            continue
        value = row.get("drawers_opened_to_success")
        if value is not None:
            grouped[str(int(row["drawer_count"]))][str(row["arm"])].append(float(value))
    return {count: {arm: summary_stats(values) for arm, values in sorted(arms.items())} for count, arms in sorted(grouped.items(), key=lambda item: int(item[0]))}


def summary_stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0, "mean": 0.0, "ci95": 0.0}
    avg = sum(values) / len(values)
    if len(values) < 2:
        ci95 = 0.0
    else:
        variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
        ci95 = 1.96 * math.sqrt(variance) / math.sqrt(len(values))
    return {"n": len(values), "mean": round(avg, 6), "ci95": round(ci95, 6)}


def gridworld_curve(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_count in sorted(summary["curve_by_drawer_count"], key=lambda value: int(value)):
        row = summary["curve_by_drawer_count"][raw_count]
        metrics = row["metrics_by_arm"]
        pairwise = row["pairwise"]
        rows.append(
            {
                "drawer_count": int(raw_count),
                "clean_drawers": float(metrics["memory_clean"]["mean_drawers_opened_to_success"]),
                "search_drawers": float(metrics["no_memory_search"]["mean_drawers_opened_to_success"]),
                "wrong_binding_drawers": float(metrics["memory_wrong_binding"]["mean_drawers_opened_to_success"]),
                "clean_minus_search_drawers": float(pairwise["clean_minus_no_memory_drawers_opened"]),
                "wrong_binding_minus_search_steps": float(pairwise["wrong_binding_minus_no_memory_steps"]),
            }
        )
    return rows


def write_autoencode_gap_curve(metrics: dict[str, Any], path: Path) -> None:
    auto = metrics["autoencode"]["hand_coded_series"]
    series = [
        ("off", auto["off"], "#6b7280"),
        ("hand-coded clean", auto["clean"], "#2563eb"),
        ("content corrupt", auto["content_corrupt"], "#dc2626"),
        ("shuffled", auto["shuffled"], "#f59e0b"),
    ]
    write_line_svg(
        path,
        title="Autoencode recall success by gap",
        x_label="Query gap",
        y_label="Success rate",
        x_values=[row["gap"] for row in auto["off"]],
        series=[(name, [(row["gap"], row["success_rate"]) for row in rows], color) for name, rows, color in series],
        y_min=0.0,
        y_max=1.0,
    )


def write_autoencode_learned_read(metrics: dict[str, Any], path: Path) -> None:
    auto = metrics["autoencode"]["learned_series"]
    series = [
        ("hand-coded clean", auto["hand_coded_clean"], "#2563eb"),
        ("learned clean", auto["learned_clean"], "#16a34a"),
        ("content corrupt", auto["content_corrupt"], "#dc2626"),
        ("order corrupt", auto["order_corrupt"], "#7c3aed"),
        ("shuffled", auto["shuffled"], "#f59e0b"),
    ]
    write_line_svg(
        path,
        title="Autoencode learned read versus controls",
        x_label="Query gap",
        y_label="Success rate",
        x_values=[row["gap"] for row in auto["learned_clean"]],
        series=[(name, [(row["gap"], row["success_rate"]) for row in rows], color) for name, rows, color in series],
        y_min=0.0,
        y_max=1.0,
    )


def write_count_recall_accuracy(metrics: dict[str, Any], path: Path) -> None:
    arms = metrics["count_recall"]["arms"]
    values = [
        ("no memory", arms["no_memory"], "#6b7280"),
        ("clean", arms["clean"], "#16a34a"),
        ("shuffled", arms["shuffled"], "#f59e0b"),
        ("wrong binding", arms["wrong_binding"], "#dc2626"),
        ("stale", arms["stale"], "#7c3aed"),
        ("oracle", arms["oracle"], "#2563eb"),
    ]
    write_bar_svg(path, title="CountRecall query accuracy", y_label="Accuracy", values=values, y_max=1.0)


def write_gridworld_drawer_sweep(metrics: dict[str, Any], path: Path) -> None:
    rows = metrics["gridworld"]["curve"]
    write_line_svg(
        path,
        title="Gridworld retrieval cost by drawer count",
        x_label="Drawer count",
        y_label="Drawers opened before success",
        x_values=[row["drawer_count"] for row in rows],
        series=[
            ("clean", [(row["drawer_count"], row["clean_drawers"]) for row in rows], "#16a34a"),
            ("systematic search", [(row["drawer_count"], row["search_drawers"]) for row in rows], "#6b7280"),
            ("wrong binding", [(row["drawer_count"], row["wrong_binding_drawers"]) for row in rows], "#dc2626"),
        ],
        y_min=0.0,
        y_max=max(row["wrong_binding_drawers"] for row in rows) + 1.0,
    )


def write_line_svg(
    path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    x_values: list[int],
    series: list[tuple[str, list[tuple[float, float]], str]],
    y_min: float,
    y_max: float,
) -> None:
    width, height = 780, 460
    left, right, top, bottom = 76, 220, 54, 78
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_min, x_max = min(x_values), max(x_values)

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w if x_max != x_min else left + plot_w / 2

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    parts = svg_header(width, height, title)
    parts.extend(axes(left, top, plot_w, plot_h, x_label, y_label))
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0] if y_max <= 1.01 else nice_ticks(y_max):
        y = sy(tick)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" class="tick">{tick:.2g}</text>')
    for value in x_values:
        x = sx(value)
        parts.append(f'<text x="{x:.2f}" y="{top + plot_h + 24}" text-anchor="middle" class="tick">{value}</text>')
    legend_y = top + 12
    for idx, (name, rows, color) in enumerate(series):
        points = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in rows)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        for x, y in rows:
            parts.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="4" fill="{color}"/>')
        ly = legend_y + idx * 24
        parts.append(f'<line x1="{width - right + 32}" y1="{ly}" x2="{width - right + 54}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{width - right + 62}" y="{ly + 4}" class="legend">{escape(name)}</text>')
    parts.append("</svg>\n")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_bar_svg(path: Path, *, title: str, y_label: str, values: list[tuple[str, float, str]], y_max: float) -> None:
    width, height = 760, 430
    left, right, top, bottom = 78, 30, 54, 112
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sy(value: float) -> float:
        return top + (y_max - value) / y_max * plot_h

    parts = svg_header(width, height, title)
    parts.extend(axes(left, top, plot_w, plot_h, "", y_label))
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = sy(tick)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" class="tick">{tick:.2g}</text>')
    gap = 16
    bar_w = (plot_w - gap * (len(values) + 1)) / len(values)
    for idx, (name, value, color) in enumerate(values):
        x = left + gap + idx * (bar_w + gap)
        y = sy(value)
        h = top + plot_h - y
        parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="{color}"/>')
        parts.append(f'<text x="{x + bar_w / 2:.2f}" y="{y - 8:.2f}" text-anchor="middle" class="value">{value:.3f}</text>')
        parts.append(f'<text x="{x + bar_w / 2:.2f}" y="{top + plot_h + 28}" text-anchor="middle" class="tick">{escape(name)}</text>')
    parts.append("</svg>\n")
    path.write_text("\n".join(parts), encoding="utf-8")


def axes(left: int, top: int, plot_w: int, plot_h: int, x_label: str, y_label: str) -> list[str]:
    return [
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827"/>',
        f'<text x="{left + plot_w / 2:.2f}" y="{top + plot_h + 58}" text-anchor="middle" class="axis">{escape(x_label)}</text>',
        f'<text x="22" y="{top + plot_h / 2:.2f}" transform="rotate(-90 22 {top + plot_h / 2:.2f})" text-anchor="middle" class="axis">{escape(y_label)}</text>',
    ]


def svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        "<style>",
        "text { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: #111827; }",
        ".title { font-size: 20px; font-weight: 700; }",
        ".axis { font-size: 13px; font-weight: 600; fill: #374151; }",
        ".tick { font-size: 11px; fill: #4b5563; }",
        ".legend { font-size: 12px; fill: #111827; }",
        ".value { font-size: 11px; fill: #111827; }",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="28" y="32" class="title">{escape(title)}</text>',
    ]


def nice_ticks(y_max: float) -> list[float]:
    step = max(1.0, round(y_max / 4))
    return [idx * step for idx in range(int(y_max // step) + 2)]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing source artifact: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return data


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing source artifact: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected JSONL objects at {path}")
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
