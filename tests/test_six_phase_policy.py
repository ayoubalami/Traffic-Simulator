import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pygame

from config import CONFIG, build_runtime_config
from simulation.evaluation import (
    calculate_fitness,
    calculate_six_phase_fitness,
    evaluate_six_phase_policy,
    evaluate_six_phase_policy_across_seeds,
)
from simulation.metrics import Metrics
from simulation.simulation import Simulation
from simulation.six_phase_neuroevolution import SixPhasePolicy
from simulation.traffic_light import SixPhaseTrafficLightController


class SixPhasePolicyTests(unittest.TestCase):
    def setUp(self):
        self.config = build_runtime_config(CONFIG)

    def test_controller_can_activate_one_approach_only(self):
        timing = self.config["traffic_lights"]
        timing["min_green_duration_s"] = 0.1
        timing["max_green_duration_s"] = 2.0
        timing["yellow_duration_s"] = 0.1
        timing["all_red_clearance_duration_s"] = 0.0
        controller = SixPhaseTrafficLightController(self.config)
        controller.set_phase_selector(
            lambda active_phase, available_phases: "north_only"
        )

        controller.update(0.11)
        self.assertEqual(controller.phase_state, "yellow")
        controller.update(0.11)
        self.assertEqual(controller.phase_state, "all_red")
        controller.update(0.01)

        self.assertEqual(controller.active_phase, "north_only")
        self.assertEqual(controller.get_state("north"), "green")
        for direction in ("south", "east", "west"):
            self.assertEqual(controller.get_state(direction), "red")

    def test_policy_selects_only_from_available_phases(self):
        policy = SixPhasePolicy(
            [0.0] * SixPhasePolicy.genome_size,
            duration_bounds_s=(1.0, 10.0),
        )

        selected = policy.select_phase({}, ("west_only",))

        self.assertEqual(selected, "west_only")

    def test_policy_exposes_six_normalized_output_probabilities(self):
        policy = SixPhasePolicy(
            [0.0] * SixPhasePolicy.genome_size,
            duration_bounds_s=(1.0, 10.0),
        )

        probabilities = policy.predict_phase_probabilities({})

        self.assertEqual(set(probabilities), set(policy.last_phase_probabilities))
        self.assertEqual(len(probabilities), 6)
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)
        for probability in probabilities.values():
            self.assertAlmostEqual(probability, 1.0 / 6.0)

    def test_turning_stuck_metrics_count_event_and_duration(self):
        metrics = Metrics(self.config)
        vehicle = SimpleNamespace(
            current_speed=0.0,
            last_deceleration_mps2=0.0,
            comfortable_deceleration_mps2=3.0,
            stopped=True,
            cleared_intersection=False,
            road_direction="north",
            turning=True,
        )

        metrics.update([vehicle], 0.5)
        metrics.update([vehicle], 0.5)
        summary = metrics.get_summary()

        self.assertEqual(summary["turning_stuck_events"], 1)
        self.assertEqual(summary["turning_stuck_vehicles"], 1)
        self.assertAlmostEqual(summary["total_turning_stuck_time"], 1.0)
        self.assertEqual(summary["current_turning_vehicles_stuck"], 1)

    def test_turning_penalty_applies_only_to_six_phase_fitness(self):
        metrics = {
            "throughput": 1.0,
            "avg_wait_time": 0.0,
            "avg_active_wait_time": 0.0,
            "max_wait_time": 0.0,
            "queue_lengths": {},
            "total_turning_stuck_time": 2.0,
            "turning_stuck_events": 1,
        }
        base = calculate_fitness(metrics)
        six_phase = calculate_six_phase_fitness(
            metrics,
            six_phase_config={
                "turning_stuck_time_penalty": 20.0,
                "turning_stuck_event_penalty": 25.0,
            },
        )

        self.assertAlmostEqual(base, 100.0)
        self.assertAlmostEqual(six_phase, 35.0)

    def test_gridlock_receives_heavy_fitness_penalty(self):
        metrics = {
            "throughput": 1.0,
            "avg_wait_time": 0.0,
            "avg_active_wait_time": 0.0,
            "max_wait_time": 0.0,
            "queue_lengths": {},
            "gridlock_detected": 1,
        }

        fitness = calculate_six_phase_fitness(
            metrics,
            six_phase_config={"gridlock_penalty": 100000.0},
        )

        self.assertAlmostEqual(fitness, -99900.0)

    def test_intersection_stuck_count_uses_physical_position_and_speed(self):
        simulation = object.__new__(Simulation)
        simulation.config = self.config
        center = (
            self.config["window"]["width"] // 2,
            self.config["window"]["height"] // 2,
        )
        simulation.vehicles = [
            SimpleNamespace(
                current_speed=0.0,
                get_rect=lambda: pygame.Rect(center[0] - 5, center[1] - 5, 10, 10),
            ),
            SimpleNamespace(
                current_speed=0.0,
                get_rect=lambda: pygame.Rect(0, 0, 10, 10),
            ),
            SimpleNamespace(
                current_speed=100.0,
                get_rect=lambda: pygame.Rect(center[0] - 5, center[1] - 5, 10, 10),
            ),
        ]

        count = simulation.count_stuck_vehicles_in_intersection(0.5)

        self.assertEqual(count, 1)

    def test_gridlock_stops_six_phase_evaluation_early(self):
        class FakeMetrics:
            def get_summary(self):
                return {
                    "throughput": 0,
                    "avg_wait_time": 0.0,
                    "avg_active_wait_time": 0.0,
                    "max_wait_time": 0.0,
                    "queue_lengths": {},
                }

        class GridlockedSimulation:
            def __init__(self, *args, **kwargs):
                self.metrics = FakeMetrics()

            def update(self, dt):
                pass

            def count_stuck_vehicles_in_intersection(self, speed_threshold_mps):
                return 3

        config = self.config.copy()
        config["six_phase_fitness"] = {
            "gridlock_min_stuck_vehicles": 3,
            "gridlock_speed_threshold_mps": 0.5,
            "gridlock_persistence_s": 2.0,
            "gridlock_penalty": 100000.0,
        }
        policy = SimpleNamespace(select_phase=lambda *args: "ns")

        with patch("simulation.evaluation.Simulation", GridlockedSimulation):
            result = evaluate_six_phase_policy(
                config,
                policy,
                duration_s=60.0,
                timestep_s=1.0,
                speed_factor=1.0,
            )

        self.assertTrue(result["terminated_early"])
        self.assertEqual(result["termination_reason"], "intersection_gridlock")
        self.assertAlmostEqual(result["metrics"]["evaluation_elapsed_s"], 2.0)
        self.assertLess(result["fitness"], -99000.0)

    def test_gridlock_skips_remaining_candidate_seeds(self):
        gridlocked_result = {
            "fitness": -100000.0,
            "metrics": {
                name: 0.0
                for name in (
                    "throughput",
                    "avg_wait_time",
                    "avg_active_wait_time",
                    "avg_travel_time",
                    "max_wait_time",
                    "active_vehicles",
                    "avg_pedestrian_wait_time",
                    "avg_active_pedestrian_wait_time",
                    "max_pedestrian_wait_time",
                    "hard_braking_events",
                    "hard_braking_vehicles",
                    "hard_braking_vehicle_rate",
                    "max_deceleration_mps2",
                    "max_braking_intensity",
                    "total_excess_braking_intensity",
                    "avg_excess_braking_intensity_per_vehicle",
                    "turning_stuck_events",
                    "turning_stuck_vehicles",
                    "turning_stuck_vehicle_rate",
                    "total_turning_stuck_time",
                    "max_turning_vehicles_stuck",
                    "gridlock_detected",
                    "max_intersection_stuck_vehicles",
                    "evaluation_elapsed_s",
                )
            },
            "random_seed": 1,
            "terminated_early": True,
            "termination_reason": "intersection_gridlock",
        }
        gridlocked_result["metrics"]["gridlock_detected"] = 1.0
        policy = SimpleNamespace(select_phase=lambda *args: "ns")

        with patch(
            "simulation.evaluation.evaluate_six_phase_policy",
            return_value=gridlocked_result,
        ) as evaluate:
            result = evaluate_six_phase_policy_across_seeds(
                self.config,
                policy,
                seeds=(1, 2, 3),
            )

        evaluate.assert_called_once()
        self.assertEqual(result["evaluated_seeds"], (1,))


if __name__ == "__main__":
    unittest.main()
