import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from simulation.evaluation import (
    evaluate_fixed_time_policy,
    evaluate_fixed_time_policy_across_seeds,
)


class _FakeMetrics:
    def __init__(self, summary=None):
        self._summary = summary or {"throughput_rate": 0.5}

    def get_summary(self):
        return dict(self._summary)


class _FreeFlowSimulation:
    instances = []

    def __init__(self, config, **kwargs):
        self.config = config
        self.kwargs = kwargs
        self.metrics = _FakeMetrics()
        self.updated_s = 0.0
        self.__class__.instances.append(self)

    def update(self, dt):
        self.updated_s += dt

    def count_stuck_vehicles_in_intersection(self, speed_threshold_mps):
        return 0


class _GridlockedSimulation(_FreeFlowSimulation):
    def count_stuck_vehicles_in_intersection(self, speed_threshold_mps):
        return 3


class FixedTimeEvaluationTests(unittest.TestCase):
    def setUp(self):
        _FreeFlowSimulation.instances.clear()
        _GridlockedSimulation.instances.clear()
        self.plan = SimpleNamespace(
            name="equal_split",
            control_scope="vehicles_only",
        )
        self.config = {
            "road_users": {"pedestrians_enabled": False},
            "simulation": {
                "arrival_rates_per_s": {
                    "north": 0.1,
                    "south": 0.1,
                    "east": 0.1,
                    "west": 0.1,
                }
            },
            "fitness": {"throughput_rate_reward": 100.0},
            "six_phase_fitness": {
                "gridlock_min_stuck_vehicles": 3,
                "gridlock_speed_threshold_mps": 0.5,
                "gridlock_persistence_s": 2.0,
                "gridlock_penalty": 100000.0,
            },
        }

    def test_fixed_plan_uses_profile_without_neural_callbacks(self):
        original = deepcopy(self.config)
        profile = {
            "name": "east_rush",
            "arrival_rates_per_s": {
                "north": 0.05,
                "south": 0.05,
                "east": 0.8,
                "west": 0.1,
            },
            "left_turn_chance": 0.35,
            "emergency_vehicle_spawn_chance": 0.02,
        }

        with patch("simulation.evaluation.Simulation", _FreeFlowSimulation):
            result = evaluate_fixed_time_policy(
                self.config,
                self.plan,
                duration_s=2.0,
                timestep_s=1.0,
                random_seed=17,
                traffic_profile=profile,
            )

        simulation = _FreeFlowSimulation.instances[-1]
        self.assertIs(simulation.kwargs["fixed_time_plan"], self.plan)
        self.assertEqual(simulation.kwargs["random_seed"], 17)
        self.assertNotIn("phase_selector", simulation.kwargs)
        self.assertNotIn("movement_score_provider", simulation.kwargs)
        self.assertEqual(
            simulation.config["simulation"]["arrival_rates_per_s"],
            profile["arrival_rates_per_s"],
        )
        self.assertEqual(
            simulation.config["simulation"]["left_turn_chance"],
            0.35,
        )
        self.assertEqual(self.config, original)
        self.assertEqual(result["traffic_profile"], "east_rush")
        self.assertEqual(result["fixed_time_plan"], "equal_split")
        self.assertEqual(result["metrics"]["evaluation_elapsed_s"], 2.0)
        self.assertFalse(result["terminated_early"])

    def test_fixed_plan_uses_shared_gridlock_termination_and_fitness(self):
        with patch("simulation.evaluation.Simulation", _GridlockedSimulation):
            result = evaluate_fixed_time_policy(
                self.config,
                self.plan,
                duration_s=10.0,
                timestep_s=1.0,
            )

        self.assertTrue(result["terminated_early"])
        self.assertEqual(result["termination_reason"], "intersection_gridlock")
        self.assertEqual(result["metrics"]["evaluation_elapsed_s"], 2.0)
        self.assertEqual(result["metrics"]["gridlock_remaining_time_s"], 8.0)
        self.assertLess(result["fitness"], -99000.0)

    def test_across_seeds_uses_full_profile_seed_matrix(self):
        config = deepcopy(self.config)
        config["six_phase_training"] = {
            "traffic_profiles": (
                {"name": "balanced"},
                {"name": "east_rush"},
            )
        }
        config["six_phase_fitness"][
            "abort_remaining_seeds_on_gridlock"
        ] = False
        calls = []

        def evaluate_one(*args, **kwargs):
            profile = kwargs["traffic_profile"]
            seed = kwargs["random_seed"]
            calls.append((profile["name"], seed))
            return {
                "fitness": float(seed),
                "metrics": {},
                "random_seed": seed,
                "traffic_profile": profile["name"],
                "terminated_early": False,
                "termination_reason": None,
            }

        with patch(
            "simulation.evaluation.evaluate_fixed_time_policy",
            side_effect=evaluate_one,
        ):
            result = evaluate_fixed_time_policy_across_seeds(
                config,
                self.plan,
                seeds=(3, 7),
            )

        expected = (
            ("balanced", 3),
            ("balanced", 7),
            ("east_rush", 3),
            ("east_rush", 7),
        )
        self.assertEqual(tuple(calls), expected)
        self.assertEqual(result["requested_scenarios"], expected)
        self.assertEqual(result["evaluated_scenarios"], expected)
        self.assertEqual(result["requested_scenario_count"], 4)
        self.assertEqual(result["evaluated_scenario_count"], 4)
        self.assertEqual(result["skipped_scenario_count"], 0)
        self.assertAlmostEqual(result["mean_fitness"], 5.0)

    def test_explicit_no_abort_keeps_requested_scenario_pairs_comparable(self):
        scenario_pairs = (
            ({"name": "first"}, 1),
            ({"name": "second"}, 2),
        )
        calls = []

        def evaluate_one(*args, **kwargs):
            profile = kwargs["traffic_profile"]
            seed = kwargs["random_seed"]
            calls.append((profile["name"], seed))
            return {
                "fitness": -100000.0,
                "metrics": {"gridlock_detected": 1.0},
                "random_seed": seed,
                "traffic_profile": profile["name"],
                "terminated_early": True,
                "termination_reason": "intersection_gridlock",
            }

        with patch(
            "simulation.evaluation.evaluate_fixed_time_policy",
            side_effect=evaluate_one,
        ):
            result = evaluate_fixed_time_policy_across_seeds(
                self.config,
                self.plan,
                scenario_pairs=scenario_pairs,
                abort_remaining_seeds_on_gridlock=False,
            )

        self.assertEqual(tuple(calls), (("first", 1), ("second", 2)))
        self.assertEqual(result["evaluated_scenario_count"], 2)
        self.assertEqual(result["skipped_scenario_count"], 0)
        self.assertTrue(result["candidate_rejected"])


if __name__ == "__main__":
    unittest.main()
