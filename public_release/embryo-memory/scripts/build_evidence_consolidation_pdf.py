#!/usr/bin/env python
"""Build a PDF version of the evidence consolidation artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from reportlab.platypus import Flowable
except ModuleNotFoundError:
    class Flowable:  # type: ignore[no-redef]
        pass


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = ROOT / "docs" / "assets" / "evidence_consolidation" / "evidence_metrics.json"
DEFAULT_OUT = ROOT / "docs" / "evidence_consolidation.pdf"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", default=str(DEFAULT_METRICS))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    metrics = read_json(Path(args.metrics))
    build_pdf(metrics, Path(args.out))
    print(json.dumps({"out": str(Path(args.out).resolve())}, sort_keys=True))
    return 0


def build_pdf(metrics: dict[str, Any], out: Path) -> None:
    try:
        from reportlab.graphics.shapes import Circle, Drawing, Line, PolyLine, Rect, String
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            Flowable,
            Image,
            KeepTogether,
            ListFlowable,
            ListItem,
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit("Install reportlab to build the PDF: python3 -m pip install --user reportlab") from exc

    out.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(out),
        pagesize=letter,
        rightMargin=0.72 * inch,
        leftMargin=0.72 * inch,
        topMargin=0.66 * inch,
        bottomMargin=0.66 * inch,
        title="Explicit Decoupled Memory for Sequential Agents",
        author="Embryo",
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER, fontSize=22, leading=27, spaceAfter=12))
    styles.add(ParagraphStyle(name="Subtitle", parent=styles["BodyText"], alignment=TA_CENTER, fontSize=10.5, leading=14, textColor=colors.HexColor("#4b5563"), spaceAfter=16))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading1"], fontSize=15, leading=18, spaceBefore=12, spaceAfter=8))
    styles.add(ParagraphStyle(name="Subsection", parent=styles["Heading2"], fontSize=12.5, leading=15, spaceBefore=8, spaceAfter=6))
    styles.add(ParagraphStyle(name="BodyTight", parent=styles["BodyText"], fontSize=9.8, leading=13, alignment=TA_LEFT, spaceAfter=6))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8.2, leading=10.2, textColor=colors.HexColor("#374151")))
    styles.add(ParagraphStyle(name="CodeBox", parent=styles["Code"], fontSize=8.8, leading=11, backColor=colors.HexColor("#f3f4f6"), borderColor=colors.HexColor("#d1d5db"), borderWidth=0.6, borderPadding=7, spaceAfter=8))

    story: list[Any] = []
    story.append(Paragraph("Explicit Decoupled Memory for Sequential Agents", styles["TitleCenter"]))
    story.append(Paragraph("Evidence consolidation v0", styles["Subtitle"]))
    story.append(Paragraph("This artifact consolidates the controlled evidence currently reproducible in the Embryo repository for explicit, decoupled memory in sequential agents.", styles["BodyTight"]))
    story.append(Paragraph("Sized Claim", styles["Section"]))
    story.append(Paragraph("An explicit decoupled memory store can be written, learned, retrieved, and injected into a policy decision to improve controlled-task recall accuracy and retrieval efficiency on tasks where correct stored content is required.", styles["CodeBox"]))
    story.append(bullets([
        "explicit memory recovers information beyond a measured recurrent baseline falloff;",
        "a learned associative reader recovers the hand-coded memory ceiling;",
        "recalled content can drive an action-level decision;",
        "recalled location content can drive a goal-conditioned navigator, making retrieval cheaper as the number of candidate locations grows.",
    ], styles))
    story.append(Paragraph("Non-Claims", styles["Section"]))
    story.append(bullets([
        "no learned write selection or full episodic memory management;",
        "no portability to external frozen backbones;",
        "no cross-episode memory;",
        "no world-like perception or high-dimensional visual grounding;",
        "no reinforcement-learning reward improvement;",
        "no Crafter competence or general embodied-agent competence.",
    ], styles))
    story.append(Paragraph("Methodological Spine", styles["Section"]))
    story.append(bullets([
        "Fair baselines: task-trained recurrence or systematic search, not blind floors.",
        "Corruption controls: shuffled, wrong-binding, stale, content-corrupt, or order-corrupt arms.",
        "Crossover or scaling: effect appears where the baseline fails or grows with the controlled variable.",
        "Evaluability gates: contamination checks and reachability before effect interpretation.",
        "Anti-selection: tuning and final reporting stay separated.",
    ], styles))
    story.append(Paragraph("Mechanism Overview", styles["Section"]))
    story.append(MemoryFlowDiagram())
    story.append(PageBreak())

    add_autoencode_sections(story, styles, metrics)
    story.append(PageBreak())
    add_count_recall_section(story, styles, metrics)
    story.append(PageBreak())
    add_gridworld_section(story, styles, metrics)
    story.append(PageBreak())
    add_boundary_sections(story, styles)

    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)


def add_autoencode_sections(story: list[Any], styles: dict[str, Any], metrics: dict[str, Any]) -> None:
    from reportlab.platypus import Paragraph

    auto = metrics["autoencode"]
    hand_coded = auto["hand_coded_series"]
    learned = auto["learned_series"]
    story.append(Paragraph("Evidence Rung 1: Recall Past Recurrent Saturation", styles["Section"]))
    story.append(Paragraph("Question: Can explicit memory recall information after a fair recurrent baseline falls off?", styles["BodyTight"]))
    story.append(Paragraph("Task: POPGym Autoencode Easy.", styles["BodyTight"]))
    story.append(Paragraph("Source artifacts: runs/popgym_autoencode_recurrent/popgym_autoencode_summary.json and runs/popgym_autoencode_recurrent/memory_curve.json.", styles["Small"]))
    story.append(LineChartFlowable(
        title="Autoencode recall success by gap",
        x_label="Query gap",
        y_label="Success rate",
        series=[
            ("off", [(row["gap"], row["success_rate"]) for row in hand_coded["off"]], "#6b7280"),
            ("hand-coded clean", [(row["gap"], row["success_rate"]) for row in hand_coded["clean"]], "#2563eb"),
            ("content corrupt", [(row["gap"], row["success_rate"]) for row in hand_coded["content_corrupt"]], "#dc2626"),
            ("shuffled", [(row["gap"], row["success_rate"]) for row in hand_coded["shuffled"]], "#f59e0b"),
        ],
        y_min=0.0,
        y_max=1.0,
    ))
    rows = [["Gap", "No memory", "Hand-coded clean", "Content corrupt", "Shuffled"]]
    for gap in [1, 4, 10, 27, 52]:
        rows.append([
            str(gap),
            f"{lookup(hand_coded, 'off', gap):.6g}",
            f"{lookup(hand_coded, 'clean', gap):.6g}",
            f"{lookup(hand_coded, 'content_corrupt', gap):.6g}",
            f"{lookup(hand_coded, 'shuffled', gap):.6g}",
        ])
    story.append(data_table(rows))
    story.append(Paragraph("Sample count: 128 evaluated sequences per arm and gap.", styles["Small"]))
    story.append(Paragraph("Supports: explicit memory fills a measured recurrence gap. Does not support learned write selection or task performance beyond recall itself.", styles["BodyTight"]))

    story.append(Paragraph("Evidence Rung 2: Learned Associative Read", styles["Section"]))
    story.append(Paragraph("Question: Can a learned memory reader recover the hand-coded ceiling?", styles["BodyTight"]))
    story.append(Paragraph("Source artifacts: runs/popgym_autoencode_recurrent/learned_memory_curve.json and runs/popgym_autoencode_recurrent/learned_memory_checkpoint/manifest.json.", styles["Small"]))
    story.append(LineChartFlowable(
        title="Autoencode learned-read success by gap",
        x_label="Query gap",
        y_label="Success rate",
        series=[
            ("hand-coded clean", [(row["gap"], row["success_rate"]) for row in learned["hand_coded_clean"]], "#2563eb"),
            ("learned clean", [(row["gap"], row["success_rate"]) for row in learned["learned_clean"]], "#16a34a"),
            ("content corrupt", [(row["gap"], row["success_rate"]) for row in learned["content_corrupt"]], "#dc2626"),
            ("order corrupt", [(row["gap"], row["success_rate"]) for row in learned["order_corrupt"]], "#7c3aed"),
            ("shuffled", [(row["gap"], row["success_rate"]) for row in learned["shuffled"]], "#f59e0b"),
        ],
        y_min=0.0,
        y_max=1.0,
    ))
    arm = auto["learned_summary_by_arm"]
    story.append(data_table([
        ["Arm", "Success"],
        ["Hand-coded clean", f"{arm['autoencode_hand_coded_memory_clean']['success_rate']:.6g}"],
        ["Learned clean", f"{arm['autoencode_learned_memory_clean']['success_rate']:.6g}"],
        ["No memory", f"{arm['autoencode_memory_off']['success_rate']:.6g}"],
        ["Content corrupt", f"{arm['autoencode_learned_memory_content_corrupt']['success_rate']:.6g}"],
        ["Order corrupt", f"{arm['autoencode_learned_memory_order_corrupt']['success_rate']:.6g}"],
        ["Shuffled", f"{arm['autoencode_learned_memory_shuffled']['success_rate']:.6g}"],
    ]))
    story.append(Paragraph("Sample count: 1152 evaluated rows per aggregate arm.", styles["Small"]))
    story.append(Paragraph("Supports: the read can be learned and preserves the corruption signature. Does not support learned memory management; the write rule still stores the needed sequence.", styles["BodyTight"]))


def add_count_recall_section(story: list[Any], styles: dict[str, Any], metrics: dict[str, Any]) -> None:
    from reportlab.platypus import Paragraph

    count = metrics["count_recall"]
    arms = count["arms"]
    story.append(Paragraph("Evidence Rung 3: Action-Level Injection", styles["Section"]))
    story.append(Paragraph("Question: Can recalled content be injected into a decision when the correct action is the recalled value?", styles["BodyTight"]))
    story.append(Paragraph("Task: POPGym CountRecall Hard.", styles["BodyTight"]))
    story.append(Paragraph("Source artifacts: runs/popgym_count_recall_injection/popgym_count_recall_summary.json and runs/popgym_count_recall_injection/metrics_by_arm.json.", styles["Small"]))
    story.append(BarChartFlowable(
        title="CountRecall query accuracy",
        y_label="Accuracy",
        values=[
            ("no memory", arms["no_memory"], "#6b7280"),
            ("clean", arms["clean"], "#16a34a"),
            ("shuffled", arms["shuffled"], "#f59e0b"),
            ("wrong binding", arms["wrong_binding"], "#dc2626"),
            ("stale", arms["stale"], "#7c3aed"),
            ("oracle", arms["oracle"], "#2563eb"),
        ],
        y_max=1.0,
    ))
    arm_metrics = count_recall_arm_metrics(count)
    ci = count["episode_ci"]
    story.append(data_table([
        ["Arm", "Query accuracy (95% CI)", "Mean reward", "Invalid action rate"],
        ["No memory", ci_row(ci, "count_recall_no_memory"), metric_row(arm_metrics, "count_recall_no_memory", "mean_reward_sum"), metric_row(arm_metrics, "count_recall_no_memory", "invalid_action_rate")],
        ["Clean memory", ci_row(ci, "count_recall_memory_clean"), metric_row(arm_metrics, "count_recall_memory_clean", "mean_reward_sum"), metric_row(arm_metrics, "count_recall_memory_clean", "invalid_action_rate")],
        ["Shuffled", ci_row(ci, "count_recall_memory_shuffled"), metric_row(arm_metrics, "count_recall_memory_shuffled", "mean_reward_sum"), metric_row(arm_metrics, "count_recall_memory_shuffled", "invalid_action_rate")],
        ["Wrong binding", ci_row(ci, "count_recall_memory_wrong_binding"), metric_row(arm_metrics, "count_recall_memory_wrong_binding", "mean_reward_sum"), metric_row(arm_metrics, "count_recall_memory_wrong_binding", "invalid_action_rate")],
        ["Stale", ci_row(ci, "count_recall_memory_stale"), metric_row(arm_metrics, "count_recall_memory_stale", "mean_reward_sum"), metric_row(arm_metrics, "count_recall_memory_stale", "invalid_action_rate")],
        ["Eval-only oracle", ci_row(ci, "count_recall_oracle_eval_only"), metric_row(arm_metrics, "count_recall_oracle_eval_only", "mean_reward_sum"), metric_row(arm_metrics, "count_recall_oracle_eval_only", "invalid_action_rate")],
    ]))
    receipt = count["source_receipt"]["reproduction"]
    story.append(Paragraph(
        "Named deviation: CountRecall has an integer-division/action-space off-by-one. The installed source sets Discrete(max_card_count), but the rewarded count can equal max_card_count. Source receipt: "
        f"{receipt['task']} {receipt['action_space']}, seed {receipt['seed']}, tick {receipt['tick']}, target {receipt['target_count']}, "
        f"action_space.contains(target)={receipt['action_space_contains_target']}, reward_for_target={receipt['reward_for_target']:.6g}. "
        "The probe corrects the local declared action space with count_recall_inclusive_max_count_action_space_v0; all arms use the same corrected space.",
        styles["BodyTight"],
    ))
    story.append(Paragraph("Supports: recalled content can cross from memory into a policy decision. Does not support goal-conditioned injection; here the action is the recalled count.", styles["BodyTight"]))


def add_gridworld_section(story: list[Any], styles: dict[str, Any], metrics: dict[str, Any]) -> None:
    from reportlab.platypus import Paragraph

    grid = metrics["gridworld"]
    story.append(Paragraph("Evidence Rung 4: Goal-Channel Injection", styles["Section"]))
    story.append(Paragraph("Question: Can recalled content become a navigation goal for a policy whose actions are computed over multiple steps?", styles["BodyTight"]))
    story.append(Paragraph("Task: controlled gridworld kitchen abstraction.", styles["BodyTight"]))
    story.append(Paragraph("Protocol: reveal object location, write it to agent-owned memory, occlude, query the needed object, inject the recalled drawer as a goal into a frozen goal-conditioned navigator. If a memory-directed drawer is wrong, the agent falls back to systematic sweep.", styles["BodyTight"]))
    story.append(LineChartFlowable(
        title="Gridworld retrieval cost by drawer count",
        x_label="Drawer count",
        y_label="Drawers opened before success",
        series=[
            ("clean", [(row["drawer_count"], row["clean_drawers"]) for row in grid["curve"]], "#16a34a"),
            ("systematic search", [(row["drawer_count"], row["search_drawers"]) for row in grid["curve"]], "#6b7280"),
            ("wrong binding", [(row["drawer_count"], row["wrong_binding_drawers"]) for row in grid["curve"]], "#dc2626"),
        ],
        y_min=0.0,
        y_max=max(row["wrong_binding_drawers"] for row in grid["curve"]) + 1,
    ))
    arms = grid["arms"]
    story.append(data_table([
        ["Arm", "Success", "Drawers opened", "Steps", "Wrong drawer"],
        ["Systematic search", metric_row(arms, "no_memory_search", "retrieval_success_rate"), metric_row(arms, "no_memory_search", "mean_drawers_opened_to_success"), metric_row(arms, "no_memory_search", "mean_steps_to_retrieve"), metric_row(arms, "no_memory_search", "wrong_drawer_rate")],
        ["Clean memory", metric_row(arms, "memory_clean", "retrieval_success_rate"), metric_row(arms, "memory_clean", "mean_drawers_opened_to_success"), metric_row(arms, "memory_clean", "mean_steps_to_retrieve"), metric_row(arms, "memory_clean", "wrong_drawer_rate")],
        ["Wrong binding", metric_row(arms, "memory_wrong_binding", "retrieval_success_rate"), metric_row(arms, "memory_wrong_binding", "mean_drawers_opened_to_success"), metric_row(arms, "memory_wrong_binding", "mean_steps_to_retrieve"), metric_row(arms, "memory_wrong_binding", "wrong_drawer_rate")],
        ["Eval-only oracle", metric_row(arms, "oracle_goal_eval_only", "retrieval_success_rate"), metric_row(arms, "oracle_goal_eval_only", "mean_drawers_opened_to_success"), metric_row(arms, "oracle_goal_eval_only", "mean_steps_to_retrieve"), metric_row(arms, "oracle_goal_eval_only", "wrong_drawer_rate")],
    ]))
    ci = grid["drawers_opened_ci"]
    story.append(data_table([
        ["Arm", "Drawers opened (95% CI)", "Episodes"],
        ["Systematic search", ci_row(ci, "no_memory_search"), str(ci["no_memory_search"]["n"])],
        ["Clean memory", ci_row(ci, "memory_clean"), str(ci["memory_clean"]["n"])],
        ["Wrong binding", ci_row(ci, "memory_wrong_binding"), str(ci["memory_wrong_binding"]["n"])],
    ]))
    story.append(Paragraph("Clean and oracle drawer counts have structural zero variance because correct goal recall opens exactly one drawer.", styles["Small"]))
    rows = [["Drawer count", "Clean drawers", "Search drawers", "Gap"]]
    for row in grid["curve"]:
        rows.append([str(row["drawer_count"]), f"{row['clean_drawers']:.6g}", f"{row['search_drawers']:.6g}", f"{row['clean_minus_search_drawers']:.6g}"])
    story.append(data_table(rows))
    story.append(Paragraph(f"Supports: goal-channel injection works when recall and action are separated by the policy's path computation. The memory advantage grows with search-space size. Aggregate wrong-binding step cost over systematic search is {grid['pairwise']['wrong_binding_minus_no_memory_steps']:.6g} steps because the navigator commits to the wrong recalled goal before fallback.", styles["BodyTight"]))
    story.append(Paragraph("Does not support learned write selection, cross-episode memory, real perception, or external-backbone portability.", styles["BodyTight"]))


def add_boundary_sections(story: list[Any], styles: dict[str, Any]) -> None:
    from reportlab.platypus import Paragraph

    story.append(Paragraph("Negative and Scoped-Out Diagnostics", styles["Section"]))
    story.append(bullets([
        "Crafter long-run work supports memory-evaluability and mechanism diagnostics, not Crafter competence.",
        "Closed-loop reward probes showed that behavior-cloned recurrent baselines can have high per-step teacher agreement while failing task completion.",
        "Runnable maze-like tasks are not automatically memory-necessary tasks.",
        "A tuned scalar bias can create apparent survival gains on a small evaluation set; those numbers are not memory evidence without held-out selection discipline.",
    ], styles))
    story.append(Paragraph("Current Boundary", styles["Section"]))
    story.append(Paragraph("The consolidated evidence is strongest for controlled within-episode associative recall and injection efficiency.", styles["BodyTight"]))
    story.append(Paragraph("It is not evidence yet for external-backbone portability, learned write selection, cross-episode memory, reinforcement-learning reward improvement, or world-like embodied competence.", styles["BodyTight"]))
    story.append(Paragraph("Next Keystone Choice", styles["Section"]))
    story.append(data_table([
        ["Desired claim", "Required next step"],
        ["Injection efficiency", "Review this controlled result."],
        ["External-backbone portability", "Scout and reproduce a competent external goal-conditioned backbone, then measure adapter cost and shuffle-gated memory lift."],
        ["Full episodic memory", "Build a distractor-heavy task where storing everything fails and a learned write-selection rule is required."],
        ["Reward-performance improvement", "Build or borrow a fair RL baseline first."],
    ], col_widths=[150, 340]))
    story.append(Paragraph("Reproduction Commands", styles["Section"]))
    story.append(Paragraph("python3 scripts/build_evidence_consolidation_figures.py<br/>python3 scripts/build_evidence_consolidation_pdf.py<br/>python3 -m compileall -q embryo scripts tests examples<br/>python3 -m unittest discover -s tests<br/>python3 scripts/check_release.py", styles["CodeBox"]))


class MemoryFlowDiagram(Flowable):
    def wrap(self, avail_width: float, avail_height: float) -> tuple[float, float]:
        return min(avail_width, 520), 135

    def drawOn(self, canvas: Any, x: float, y: float, _sW: float = 0) -> None:
        from reportlab.lib import colors

        labels = ["Deployable observation", "Memory write", "Explicit store", "Memory read", "Injection", "Policy", "Action", "Eval-only outcome"]
        w, h = 58, 34
        gap = 5
        cx = x
        cy = y + 72
        canvas.saveState()
        canvas.setFont("Helvetica", 6.8)
        canvas.setStrokeColor(colors.HexColor("#374151"))
        for idx, label in enumerate(labels):
            bx = cx + idx * (w + gap)
            canvas.setFillColor(colors.HexColor("#fee2e2") if idx == 7 else colors.HexColor("#eef2ff") if idx in {2, 3, 4} else colors.HexColor("#f9fafb"))
            canvas.roundRect(bx, cy, w, h, 4, stroke=1, fill=1)
            canvas.setFillColor(colors.HexColor("#111827"))
            for line_no, part in enumerate(split_label(label, 10)):
                canvas.drawCentredString(bx + w / 2, cy + 21 - line_no * 9, part)
            if idx < len(labels) - 1:
                ax = bx + w
                canvas.line(ax, cy + h / 2, ax + gap, cy + h / 2)
                canvas.line(ax + gap - 4, cy + h / 2 + 3, ax + gap, cy + h / 2)
                canvas.line(ax + gap - 4, cy + h / 2 - 3, ax + gap, cy + h / 2)
        boundary_x = cx + 7 * (w + gap) - 2
        canvas.setStrokeColor(colors.HexColor("#dc2626"))
        canvas.setDash(3, 2)
        canvas.line(boundary_x, cy - 8, boundary_x, cy + h + 8)
        canvas.setDash()
        canvas.setFillColor(colors.HexColor("#991b1b"))
        canvas.setFont("Helvetica", 7.2)
        canvas.drawString(cx, y + 38, "Actor/query inputs stop before eval-only outcome fields.")
        canvas.setFillColor(colors.HexColor("#374151"))
        canvas.setFont("Helvetica", 6.8)
        canvas.drawString(cx, y + 25, "Eval-only fields score outcomes; they are not actor, query, or router inputs.")
        canvas.restoreState()


class LineChartFlowable(Flowable):
    def __init__(self, *, title: str, x_label: str, y_label: str, series: list[tuple[str, list[tuple[float, float]], str]], y_min: float, y_max: float) -> None:
        super().__init__()
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.series = series
        self.y_min = y_min
        self.y_max = y_max
        self.width = 500
        self.height = 260

    def wrap(self, avail_width: float, avail_height: float) -> tuple[float, float]:
        self.width = min(500, avail_width)
        return self.width, self.height

    def drawOn(self, canvas: Any, x: float, y: float, _sW: float = 0) -> None:
        from reportlab.lib import colors

        left, right, top, bottom = 48, 132, 28, 42
        plot_w = self.width - left - right
        plot_h = self.height - top - bottom
        all_x = [point[0] for _, rows, _ in self.series for point in rows]
        x_min, x_max = min(all_x), max(all_x)

        def sx(value: float) -> float:
            return x + left + (value - x_min) / (x_max - x_min) * plot_w if x_max != x_min else x + left + plot_w / 2

        def sy(value: float) -> float:
            return y + bottom + (value - self.y_min) / (self.y_max - self.y_min) * plot_h

        canvas.saveState()
        canvas.setFont("Helvetica-Bold", 11)
        canvas.drawString(x, y + self.height - 14, self.title)
        canvas.setStrokeColor(colors.HexColor("#111827"))
        canvas.line(x + left, y + bottom, x + left + plot_w, y + bottom)
        canvas.line(x + left, y + bottom, x + left, y + bottom + plot_h)
        canvas.setFont("Helvetica", 7.5)
        for tick in y_ticks(self.y_max):
            ty = sy(tick)
            canvas.setStrokeColor(colors.HexColor("#e5e7eb"))
            canvas.line(x + left, ty, x + left + plot_w, ty)
            canvas.setFillColor(colors.HexColor("#4b5563"))
            canvas.drawRightString(x + left - 5, ty - 2.5, f"{tick:.2g}")
        shown = sorted(set(int(v) for v in all_x))
        for value in shown:
            tx = sx(value)
            canvas.drawCentredString(tx, y + bottom - 13, str(value))
        canvas.setFont("Helvetica", 8)
        canvas.drawCentredString(x + left + plot_w / 2, y + 8, self.x_label)
        canvas.saveState()
        canvas.translate(x + 12, y + bottom + plot_h / 2)
        canvas.rotate(90)
        canvas.drawCentredString(0, 0, self.y_label)
        canvas.restoreState()
        legend_x = x + left + plot_w + 18
        legend_y = y + bottom + plot_h - 10
        for idx, (name, rows, color) in enumerate(self.series):
            canvas.setStrokeColor(colors.HexColor(color))
            path = canvas.beginPath()
            for point_idx, (px, py) in enumerate(rows):
                if point_idx == 0:
                    path.moveTo(sx(px), sy(py))
                else:
                    path.lineTo(sx(px), sy(py))
            canvas.drawPath(path, stroke=1, fill=0)
            canvas.setFillColor(colors.HexColor(color))
            for px, py in rows:
                canvas.circle(sx(px), sy(py), 2.2, stroke=0, fill=1)
            ly = legend_y - idx * 13
            canvas.rect(legend_x, ly - 4, 7, 7, stroke=0, fill=1)
            canvas.setFillColor(colors.HexColor("#111827"))
            canvas.setFont("Helvetica", 7.2)
            canvas.drawString(legend_x + 11, ly - 2, name)
        canvas.restoreState()


class BarChartFlowable(Flowable):
    def __init__(self, *, title: str, y_label: str, values: list[tuple[str, float, str]], y_max: float) -> None:
        super().__init__()
        self.title = title
        self.y_label = y_label
        self.values = values
        self.y_max = y_max
        self.width = 500
        self.height = 250

    def wrap(self, avail_width: float, avail_height: float) -> tuple[float, float]:
        self.width = min(500, avail_width)
        return self.width, self.height

    def drawOn(self, canvas: Any, x: float, y: float, _sW: float = 0) -> None:
        from reportlab.lib import colors

        left, right, top, bottom = 44, 16, 28, 54
        plot_w = self.width - left - right
        plot_h = self.height - top - bottom

        def sy(value: float) -> float:
            return y + bottom + value / self.y_max * plot_h

        canvas.saveState()
        canvas.setFont("Helvetica-Bold", 11)
        canvas.drawString(x, y + self.height - 14, self.title)
        canvas.setStrokeColor(colors.HexColor("#111827"))
        canvas.line(x + left, y + bottom, x + left + plot_w, y + bottom)
        canvas.line(x + left, y + bottom, x + left, y + bottom + plot_h)
        canvas.setFont("Helvetica", 7.5)
        for tick in [0, 0.25, 0.5, 0.75, 1.0]:
            ty = sy(tick)
            canvas.setStrokeColor(colors.HexColor("#e5e7eb"))
            canvas.line(x + left, ty, x + left + plot_w, ty)
            canvas.setFillColor(colors.HexColor("#4b5563"))
            canvas.drawRightString(x + left - 5, ty - 2.5, f"{tick:.2g}")
        gap = 8
        bar_w = (plot_w - gap * (len(self.values) + 1)) / len(self.values)
        for idx, (name, value, color) in enumerate(self.values):
            bx = x + left + gap + idx * (bar_w + gap)
            by = y + bottom
            h = value / self.y_max * plot_h
            canvas.setFillColor(colors.HexColor(color))
            canvas.rect(bx, by, bar_w, h, stroke=0, fill=1)
            canvas.setFillColor(colors.HexColor("#111827"))
            canvas.drawCentredString(bx + bar_w / 2, by + h + 5, f"{value:.3f}")
            for line_no, part in enumerate(split_label(name, 10)):
                canvas.drawCentredString(bx + bar_w / 2, y + bottom - 14 - line_no * 9, part)
        canvas.saveState()
        canvas.translate(x + 12, y + bottom + plot_h / 2)
        canvas.rotate(90)
        canvas.drawCentredString(0, 0, self.y_label)
        canvas.restoreState()
        canvas.restoreState()


def bullets(items: list[str], styles: dict[str, Any]) -> Any:
    from reportlab.platypus import ListFlowable, ListItem, Paragraph

    return ListFlowable(
        [ListItem(Paragraph(item, styles["BodyTight"]), leftIndent=12) for item in items],
        bulletType="bullet",
        leftIndent=14,
        bulletFontSize=7,
    )


def data_table(rows: list[list[str]], col_widths: list[int] | None = None) -> Any:
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle

    table = Table(rows, colWidths=col_widths, hAlign="LEFT", repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f4f6")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("LEADING", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d1d5db")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def page_footer(canvas: Any, doc: Any) -> None:
    from reportlab.lib import colors

    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawString(doc.leftMargin, 0.38 * 72, "Embryo evidence consolidation v0")
    canvas.drawRightString(doc.pagesize[0] - doc.rightMargin, 0.38 * 72, f"Page {doc.page}")
    canvas.restoreState()


def lookup(series_by_name: dict[str, Any], series: str, gap: int) -> float:
    for row in series_by_name[series]:
        if int(row["gap"]) == int(gap):
            return float(row["success_rate"])
    raise KeyError((series, gap))


def metric_row(metrics: dict[str, Any], arm: str, key: str) -> str:
    value = metrics.get(arm, {}).get(key)
    if value is None:
        return "n/a"
    return f"{float(value):.6g}"


def count_recall_arm_metrics(count: dict[str, Any]) -> dict[str, Any]:
    source_path = ROOT / count["source"]
    if source_path.exists():
        return read_json(source_path)["metrics_by_arm"]
    arms = count["arms"]
    return {
        "count_recall_no_memory": {"query_accuracy": arms["no_memory"]},
        "count_recall_memory_clean": {"query_accuracy": arms["clean"]},
        "count_recall_memory_shuffled": {"query_accuracy": arms["shuffled"]},
        "count_recall_memory_wrong_binding": {"query_accuracy": arms["wrong_binding"]},
        "count_recall_memory_stale": {"query_accuracy": arms["stale"]},
        "count_recall_oracle_eval_only": {"query_accuracy": arms["oracle"]},
    }


def ci_row(metrics: dict[str, Any], arm: str) -> str:
    row = metrics[arm]
    return f"{float(row['mean']):.6g} +/- {float(row['ci95']):.6g}"


def split_label(label: str, max_len: int) -> list[str]:
    words = label.split()
    lines: list[str] = []
    current = ""
    for word in words:
        proposed = word if not current else f"{current} {word}"
        if len(proposed) <= max_len:
            current = proposed
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines[:3]


def y_ticks(y_max: float) -> list[float]:
    if y_max <= 1.01:
        return [0, 0.25, 0.5, 0.75, 1.0]
    step = max(1, round(y_max / 4))
    ticks = [0]
    value = step
    while value <= y_max + 0.01:
        ticks.append(value)
        value += step
    return ticks


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing source artifact: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return data


if __name__ == "__main__":
    raise SystemExit(main())
