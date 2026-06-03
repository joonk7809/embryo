from __future__ import annotations

import unittest

from embryo.core.contracts import DEFAULT_CONTRACT, contract_from_mapping
from embryo.eval.contamination import flatten_keys, scan_actor_context, validate_contract


class ContractTests(unittest.TestCase):
    def test_default_contract_has_forward_surfaces(self) -> None:
        contract = DEFAULT_CONTRACT
        self.assertEqual(contract.name, "crafter_memory_v1")
        self.assertEqual(len(contract.allowed_actor_inputs), 9)
        self.assertEqual(len(contract.allowed_query_inputs), 6)
        self.assertEqual(len(contract.allowed_router_state), 5)
        self.assertIn("facing_candidate_v1", contract.allowed_names)
        self.assertIn("reward", contract.forbidden_names_and_aliases)

    def test_contract_round_trip(self) -> None:
        payload = DEFAULT_CONTRACT.to_dict()
        loaded = contract_from_mapping(payload)
        self.assertEqual(loaded.to_dict(), payload)

    def test_contract_validation_passes(self) -> None:
        scan = validate_contract(DEFAULT_CONTRACT)
        self.assertTrue(scan["passed"])
        self.assertEqual(scan["failure_count"], 0)

    def test_scan_rejects_nested_teacher_info(self) -> None:
        scan = scan_actor_context({"raw_rgb_frame": "ok", "info": {"inventory": {"wood": 1}}})
        self.assertFalse(scan["passed"])
        self.assertIn('info["inventory"]', scan["forbidden_present"])

    def test_scan_rejects_reward_and_backend_aliases(self) -> None:
        scan = scan_actor_context({"reward": 1.0, "backend_state": {"x": 1}})
        self.assertFalse(scan["passed"])
        self.assertGreaterEqual(scan["failure_count"], 2)

    def test_scan_can_fail_unknown_inputs(self) -> None:
        loose = scan_actor_context({"raw_rgb_frame": "ok", "mystery_feature": 1})
        strict = scan_actor_context({"raw_rgb_frame": "ok", "mystery_feature": 1}, strict_unknown=True)
        self.assertTrue(loose["passed"])
        self.assertFalse(strict["passed"])
        self.assertIn("mystery_feature", strict["unknown_present"])

    def test_flatten_keys_adds_info_aliases(self) -> None:
        keys = flatten_keys({"info": {"semantic": 1, "player_pos": [0, 0]}})
        self.assertIn("info.semantic", keys)
        self.assertIn('info["semantic"]', keys)
        self.assertIn("semantic", keys)
        self.assertIn("player_pos", keys)


if __name__ == "__main__":
    unittest.main()
