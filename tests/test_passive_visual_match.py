from __future__ import annotations

import unittest

from embryo.eval.long_run import GO_PASSIVE_MATCH_EVALUABLE
from embryo.memory import (
    PassiveCueMemory,
    passive_match_action_for_side,
    passive_match_choice_visible,
    passive_match_cue_fact_from_observation,
)
from embryo.run.long_run import run_long_run_protocol
from embryo.run.long_run_arms import EpisodeState, select_action_for_arm
from embryo.runtimes import make_runtime
from embryo.runtimes.passive_visual_match import PASSIVE_VISUAL_MATCH_SPEC


class PassiveVisualMatchRuntimeTests(unittest.TestCase):
    def test_runtime_reward_requires_matching_remembered_cue(self) -> None:
        left = make_runtime("passive_visual_match", max_steps=4, seed=2)
        right = make_runtime("passive_visual_match", max_steps=4, seed=3)
        try:
            left.reset(seed=2)
            right.reset(seed=3)
            for _ in range(4):
                left_step = left.step("noop")
                right_step = right.step("noop")
            self.assertTrue(left_step.done)
            self.assertTrue(right_step.done)
            self.assertEqual(left_step.reward, 0.0)
            self.assertEqual(right_step.reward, 0.0)

            left.reset(seed=2)
            right.reset(seed=3)
            for _ in range(4):
                left_action = "choose_left" if _ == 3 else "noop"
                right_action = "choose_right" if _ == 3 else "noop"
                left_step = left.step(left_action)
                right_step = right.step(right_action)
            self.assertEqual(left_step.reward, 1.0)
            self.assertEqual(right_step.reward, 1.0)
        finally:
            left.close()
            right.close()

    def test_cue_fact_and_choice_phase_are_derived_from_rgb(self) -> None:
        runtime = make_runtime("passive_visual_match", max_steps=4, seed=3)
        try:
            step = runtime.reset(seed=3)
            cue = passive_match_cue_fact_from_observation(step.observation)
            self.assertTrue(cue.value)
            self.assertEqual(cue.metadata["side"], "right")
            runtime.step("noop")
            runtime.step("noop")
            choice_step = runtime.step("noop")
            self.assertTrue(passive_match_choice_visible(choice_step.observation))
        finally:
            runtime.close()


class PassiveVisualMatchArmTests(unittest.TestCase):
    def test_clean_recalls_cue_after_delay_and_corrupt_controls_diverge(self) -> None:
        visible = make_runtime("passive_visual_match", max_steps=16, seed=3)
        try:
            step = visible.reset(seed=3)
            blank = None
            for _ in range(15):
                blank = visible.step("noop")
            choice_observation = blank.observation  # type: ignore[union-attr]
        finally:
            visible.close()

        def decision(arm: str) -> dict:
            state = EpisodeState(seed=3, arm=arm, horizon=16, passive_match_config={"h_lstm": 8, "ttl": 32})
            first = select_action_for_arm(PASSIVE_VISUAL_MATCH_SPEC, step.observation, state, arm)
            state.observe(first["action"], first["route_mode"])
            for _ in range(13):
                hidden = {"raw_rgb_frame": choice_observation["previous_rgb_frame"], "previous_rgb_frame": None, "previous_action": "noop"}
                mid = select_action_for_arm(PASSIVE_VISUAL_MATCH_SPEC, hidden, state, arm)
                state.observe(mid["action"], mid["route_mode"])
            return select_action_for_arm(PASSIVE_VISUAL_MATCH_SPEC, choice_observation, state, arm)

        clean = decision("passive_match_clean")
        off = decision("passive_match_off")
        stale = decision("passive_match_stale")
        wrong = decision("passive_match_wrong_binding")

        self.assertTrue(clean["passive_match_ere"])
        self.assertEqual(clean["passive_match_target_action"], "choose_right")
        self.assertEqual(clean["action"], "choose_right")
        self.assertTrue(clean["passive_match_recall_consistent_action"])
        self.assertEqual(off["action"], "choose_left")
        self.assertFalse(off["passive_match_recall_active"])
        self.assertEqual(stale["action"], "choose_left")
        self.assertEqual(wrong["action"], "choose_left")
        self.assertFalse(wrong["passive_match_recall_consistent_action"])

    def test_passive_cue_memory_expires_after_ttl(self) -> None:
        runtime = make_runtime("passive_visual_match", max_steps=4, seed=2)
        try:
            step = runtime.reset(seed=2)
            memory = PassiveCueMemory(ttl=1)
            memory.update(passive_match_cue_fact_from_observation(step.observation))
            self.assertEqual(memory.entry().side, "left")  # type: ignore[union-attr]
            memory.update(passive_match_cue_fact_from_observation({"raw_rgb_frame": None}))
            memory.update(passive_match_cue_fact_from_observation({"raw_rgb_frame": None}))
            self.assertIsNone(memory.entry())
        finally:
            runtime.close()

    def test_action_mapping_uses_choice_actions(self) -> None:
        self.assertEqual(passive_match_action_for_side("left", PASSIVE_VISUAL_MATCH_SPEC.action_names), "choose_left")
        self.assertEqual(passive_match_action_for_side("right", PASSIVE_VISUAL_MATCH_SPEC.action_names), "choose_right")


class PassiveVisualMatchProtocolTests(unittest.TestCase):
    def test_protocol_clean_beats_off_and_corrupt_controls(self) -> None:
        result = run_long_run_protocol(
            {
                "runtime": {"name": "passive_visual_match", "split": "dev", "seed_start": 10000, "seed_count": 8},
                "protocol": {
                    "horizons": [16],
                    "max_episodes_per_seed": 1,
                    "detail_ticks": True,
                    "passive_match": {"h_lstm": 4, "ttl": 32, "min_ere_count": 8, "min_ere_episode_rate": 1.0},
                },
                "arms": [
                    "passive_match_off",
                    "passive_match_clean",
                    "passive_match_shuffled",
                    "passive_match_stale",
                    "passive_match_wrong_binding",
                ],
                "metrics": {"memory_grounded_score": False, "passive_match": True, "contamination": True},
            }
        )
        passive = result["summary"]["passive_match"]

        self.assertEqual(passive["status"], GO_PASSIVE_MATCH_EVALUABLE)
        self.assertEqual(passive["clean_ere_count"], 8)
        self.assertEqual(passive["summary_by_arm"]["passive_match_clean"]["success_rate_eval_only"], 1.0)
        self.assertEqual(passive["summary_by_arm"]["passive_match_off"]["success_rate_eval_only"], 0.5)
        self.assertEqual(passive["summary_by_arm"]["passive_match_shuffled"]["success_rate_eval_only"], 0.0)
        self.assertEqual(result["summary"]["contamination"]["failure_count"], 0)


if __name__ == "__main__":
    unittest.main()
