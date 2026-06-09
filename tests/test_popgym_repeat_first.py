import sys
import types
import unittest

from gymnasium import spaces

from embryo.eval.long_run import GO_POPGYM_REPEAT_FIRST_EVALUABLE, popgym_repeat_first_protocol_summary
from embryo.memory.popgym_repeat_first import RepeatFirstMemory, repeat_first_target_fact_from_observation
from embryo.run.long_run_arms import EpisodeState, select_action_for_arm
from embryo.runtimes.base import RuntimeSpec
from embryo.runtimes.popgym.adapter import POPGymRuntimeAdapter


REPEAT_SPEC = RuntimeSpec(
    name="popgym_repeat_first",
    suite="popgym",
    observation_keys=("popgym_observation", "previous_popgym_observation", "previous_action"),
    action_names=("suit_0", "suit_1", "suit_2", "suit_3"),
    noop_action="suit_0",
    action_map={"suit_0": 0, "suit_1": 1, "suit_2": 2, "suit_3": 3},
)


class PopGymRepeatFirstTests(unittest.TestCase):
    def test_memory_stores_first_observed_suit_only(self):
        memory = RepeatFirstMemory(ttl=8)
        memory.update(repeat_first_target_fact_from_observation({"popgym_observation": 2}))
        memory.update(repeat_first_target_fact_from_observation({"popgym_observation": 1}))

        entry = memory.entry()
        self.assertIsNotNone(entry)
        self.assertEqual(entry.suit, 2)
        self.assertEqual(entry.age, 1)

    def test_clean_and_corrupt_arms_diverge_on_recall_tick(self):
        def decision(arm):
            state = EpisodeState(seed=7, arm=arm, horizon=8, popgym_repeat_first_config={"h_lstm": 1, "ttl": 8})
            select_action_for_arm(REPEAT_SPEC, {"popgym_observation": 2, "previous_action": "suit_0"}, state, arm)
            return select_action_for_arm(REPEAT_SPEC, {"popgym_observation": 1, "previous_action": "suit_0"}, state, arm)

        clean = decision("popgym_repeat_first_clean")
        off = decision("popgym_repeat_first_off")
        stale = decision("popgym_repeat_first_stale")
        wrong = decision("popgym_repeat_first_wrong_binding")

        self.assertTrue(clean["popgym_repeat_first_ere"])
        self.assertEqual(clean["action"], "suit_2")
        self.assertEqual(off["action"], "suit_0")
        self.assertEqual(stale["action"], "suit_0")
        self.assertNotEqual(wrong["action"], "suit_2")
        self.assertFalse(wrong["popgym_repeat_first_recall_consistent_action"])

    def test_adapter_uses_optional_popgym_env_shape(self):
        class FakeRepeatFirstEasy:
            action_space = spaces.Discrete(4)

            def __init__(self):
                self.target = 2

            def reset(self, *, seed=None, options=None):
                self.target = 2 if int(seed or 0) % 2 == 0 else 1
                return self.target, {}

            def step(self, action):
                return 0, 1.0 if int(action) == self.target else -1.0, True, False, {}

        module = types.ModuleType("popgym.envs.repeat_first")
        module.RepeatFirstEasy = FakeRepeatFirstEasy
        old_modules = {key: sys.modules.get(key) for key in ("popgym", "popgym.envs", "popgym.envs.repeat_first")}
        try:
            sys.modules["popgym"] = types.ModuleType("popgym")
            sys.modules["popgym.envs"] = types.ModuleType("popgym.envs")
            sys.modules["popgym.envs.repeat_first"] = module

            runtime = POPGymRuntimeAdapter(task="repeat_first_easy")
            first = runtime.reset(seed=4)
            self.assertEqual(first.observation["popgym_observation"], 2)
            self.assertEqual(runtime.spec.action_names, ("suit_0", "suit_1", "suit_2", "suit_3"))
            step = runtime.step("suit_2")
            self.assertTrue(step.done)
            self.assertTrue(step.info["popgym_success"])
        finally:
            for key, value in old_modules.items():
                if value is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = value

    def test_summary_emits_repeat_first_decision_and_contrasts(self):
        ticks = []
        for arm, action, success, consistent in (
            ("popgym_repeat_first_clean", "suit_2", True, True),
            ("popgym_repeat_first_off", "suit_0", False, False),
            ("popgym_repeat_first_wrong_binding", "suit_3", False, False),
        ):
            ticks.append(
                {
                    "actor": {"arm": arm, "seed": 1, "episode_id": "e1", "episode_index": 0, "tick": 4, "action": action},
                    "metrics": {
                        "popgym_repeat_first_ere": True,
                        "popgym_repeat_first_recall_active": arm != "popgym_repeat_first_off",
                        "popgym_repeat_first_recall_consistent_action": consistent,
                        "popgym_repeat_first_fact_age": 4,
                        "popgym_repeat_first_current_matches_target": False,
                    },
                    "eval_only": {"popgym_success_eval_only": success, "reward_delta_eval_only": 1.0 if success else -1.0},
                }
            )

        summary = popgym_repeat_first_protocol_summary(
            ticks,
            protocol_manifest={
                "arms": ["popgym_repeat_first_clean", "popgym_repeat_first_off", "popgym_repeat_first_wrong_binding"],
                "protocol": {"popgym_repeat_first": {"min_ere_count": 1, "min_ere_episode_rate": 1.0}},
            },
        )

        self.assertEqual(summary["status"], GO_POPGYM_REPEAT_FIRST_EVALUABLE)
        self.assertEqual(summary["clean_ere_count"], 1)
        self.assertEqual(summary["contrasts"]["popgym_repeat_first_off"]["clean_minus_control_success_rate_eval_only"], 1.0)


if __name__ == "__main__":
    unittest.main()
