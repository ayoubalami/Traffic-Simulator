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
from simulation.six_phase_neuroevolution import (
    INPUT_FEATURE_NAMES,
    SixPhasePolicy,
    SixPhasePolicyEvolution,
)
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
        self.assertEqual(policy.last_raw_best_phase, "ns")
        self.assertEqual(policy.last_selected_phase, "west_only")
        self.assertEqual(policy.last_available_phases, ("west_only",))

    def test_controller_debug_records_policy_request_and_effective_decision(self):
        controller = SixPhaseTrafficLightController(self.config)
        observation = {
            "vehicle_counts": {
                "north": 1, "south": 0, "east": 0, "west": 0,
            },
            "queue_lengths": {"north": 1},
        }
        controller.set_phase_selector(
            lambda active_phase, available_phases: "west_only"
        )

        decision = controller._choose_phase(False, observation)

        self.assertEqual(controller.last_policy_requested_phase, "west_only")
        self.assertEqual(controller.last_available_phases, ("ns",))
        self.assertEqual(controller.last_controller_decision, "ns")
        self.assertEqual(decision, "ns")

    def test_policy_exposes_ten_normalized_main_phase_probabilities(self):
        policy = SixPhasePolicy(
            [0.0] * SixPhasePolicy.genome_size,
            duration_bounds_s=(1.0, 10.0),
        )

        probabilities = policy.predict_phase_probabilities({})

        self.assertEqual(set(probabilities), set(policy.last_phase_probabilities))
        self.assertEqual(len(probabilities), 10)
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)
        for probability in probabilities.values():
            self.assertAlmostEqual(probability, 1.0 / 10.0)

    def test_parallel_training_matches_sequential_training(self):
        options = {
            "duration_bounds_s": (1.0, 3.0),
            "population_size": 2,
            "generations": 1,
            "seeds": (1,),
            "evaluation_duration_s": 0.1,
            "speed_factor": 2.0,
            "traffic_profiles": (),
            "random_seed": 17,
        }

        sequential = SixPhasePolicyEvolution(
            self.config,
            workers=1,
            **options,
        ).run()
        parallel = SixPhasePolicyEvolution(
            self.config,
            workers=2,
            **options,
        ).run()

        self.assertEqual(
            parallel["best"]["fitness"],
            sequential["best"]["fitness"],
        )
        self.assertEqual(
            parallel["best"]["policy"].weights,
            sequential["best"]["policy"].weights,
        )
        self.assertIn(
            "avg_pre_intersection_wait_time_by_direction",
            parallel["best"]["mean_metrics"],
        )

    def test_policy_has_compact_57_input_schema(self):
        policy = SixPhasePolicy(
            [0.0] * SixPhasePolicy.genome_size,
            duration_bounds_s=(1.0, 10.0),
        )
        observation = {
            "average_speed_ratios": {
                "north": 0.25,
                "south": 0.50,
                "east": 0.75,
                "west": 1.00,
            },
            "left_red_elapsed_s": {
                "north": 15.0,
                "south": 30.0,
                "east": 45.0,
                "west": 60.0,
            },
            "right_red_elapsed_s": {
                "north": 6.0,
                "south": 12.0,
                "east": 18.0,
                "west": 24.0,
            },
            "intersection_vehicle_count": 5,
            "blocked_intersection_vehicle_count": 2,
        }

        inputs = policy._build_inputs(observation)

        self.assertEqual(SixPhasePolicy.input_size, 57)
        self.assertEqual(len(INPUT_FEATURE_NAMES), 57)
        self.assertEqual(SixPhasePolicy.genome_size, 690)
        self.assertEqual(inputs[8:12], [0.25, 0.50, 0.75, 1.00])
        self.assertEqual(inputs[36:40], [0.25, 0.50, 0.75, 1.00])
        self.assertEqual(inputs[40:44], [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(inputs[44:46], [0.5, 0.4])

    def test_observation_averages_vehicle_speed_ratio_by_approach(self):
        simulation = Simulation(self.config, random_seed=1)
        simulation.vehicles = [
            SimpleNamespace(
                cleared_intersection=False,
                road_direction="north",
                current_speed=25.0,
                speed=100.0,
                is_emergency=False,
                stopped=False,
                turning=False,
            ),
            SimpleNamespace(
                cleared_intersection=False,
                road_direction="north",
                current_speed=75.0,
                speed=100.0,
                is_emergency=False,
                stopped=False,
                turning=False,
            ),
        ]

        observation = simulation.get_signal_observation("ns")

        self.assertAlmostEqual(observation["average_speed_ratios"]["north"], 0.5)
        for direction in ("south", "east", "west"):
            self.assertEqual(observation["average_speed_ratios"][direction], 0.0)

    def test_observation_exposes_approaching_turn_demand_before_entry(self):
        simulation = Simulation(self.config, random_seed=1)
        simulation.vehicles = [
            SimpleNamespace(
                cleared_intersection=False,
                road_direction="north",
                current_speed=0.0,
                speed=100.0,
                is_emergency=False,
                stopped=True,
                turning=False,
                is_turning_vehicle=True,
                has_turned=False,
                turn_side="left",
                distance_from_stop=1000.0,
            ),
            SimpleNamespace(
                cleared_intersection=False,
                road_direction="south",
                current_speed=50.0,
                speed=100.0,
                is_emergency=False,
                stopped=False,
                turning=False,
                is_turning_vehicle=True,
                has_turned=False,
                turn_side="right",
                distance_from_stop=80.0,
            ),
        ]

        observation = simulation.get_signal_observation("ns")

        self.assertEqual(observation["approaching_left_turn_counts"]["north"], 1)
        self.assertEqual(observation["queued_left_turn_counts"]["north"], 1)
        self.assertEqual(observation["approaching_right_turn_counts"]["south"], 1)
        self.assertEqual(observation["queued_right_turn_counts"]["south"], 0)
        self.assertEqual(observation["queued_movement_counts"]["north_left"], 1)
        self.assertEqual(observation["queued_movement_counts"]["south_right"], 0)
        self.assertEqual(observation["near_stop_movement_counts"]["north_left"], 1)
        self.assertEqual(observation["near_stop_movement_counts"]["south_right"], 1)

    def test_no_turn_demand_exposes_only_paired_axis_phase(self):
        controller = SixPhaseTrafficLightController(self.config)
        observation = {
            "vehicle_counts": {
                "north": 4, "south": 3, "east": 0, "west": 0,
            },
            "queue_lengths": {},
            "approaching_left_turn_counts": {},
        }

        available = controller.get_available_phases(observation)

        self.assertEqual(available, ("ns",))

    def test_conflicting_left_turn_exposes_protected_left_phases(self):
        controller = SixPhaseTrafficLightController(self.config)
        observation = {
            "vehicle_counts": {
                "north": 4, "south": 3, "east": 0, "west": 0,
            },
            "queue_lengths": {},
            "approaching_left_turn_counts": {"north": 1},
        }

        available = controller.get_available_phases(observation)

        self.assertEqual(available, ("ns", "north_only", "north_left"))

    def test_left_turn_vehicle_obeys_only_protected_arrow(self):
        timing = self.config["traffic_lights"]
        timing["min_green_duration_s"] = 0.1
        timing["yellow_duration_s"] = 0.1
        timing["all_red_clearance_duration_s"] = 0.0
        controller = SixPhaseTrafficLightController(self.config)
        left_vehicle = SimpleNamespace(
            road_direction="north",
            is_turning_vehicle=True,
            turn_side="left",
            has_turned=False,
        )
        through_vehicle = SimpleNamespace(
            road_direction="north",
            is_turning_vehicle=False,
            turn_side=None,
            has_turned=False,
        )

        self.assertEqual(controller.active_phase, "ns")
        self.assertEqual(controller.get_vehicle_state(left_vehicle), "red")
        self.assertEqual(controller.get_vehicle_state(through_vehicle), "green")

        controller.set_phase_selector(
            lambda active_phase, available_phases: "north_left"
        )
        controller.update(0.11)
        controller.update(0.11)
        controller.update(0.01)

        self.assertEqual(controller.active_phase, "north_left")
        self.assertEqual(controller.get_state("north"), "red")
        self.assertEqual(controller.get_left_turn_state("north"), "green")
        self.assertEqual(controller.get_vehicle_state(left_vehicle), "green")
        self.assertEqual(controller.get_vehicle_state(through_vehicle), "red")

    def test_multiple_compatible_right_arrows_activate_together(self):
        controller = SixPhaseTrafficLightController(self.config)
        north_right = SimpleNamespace(
            road_direction="north",
            is_turning_vehicle=True,
            turn_side="right",
            has_turned=False,
        )
        south_right = SimpleNamespace(
            road_direction="south",
            is_turning_vehicle=True,
            turn_side="right",
            has_turned=False,
        )

        controller.set_phase_observation_provider(
            lambda: {
                "vehicle_counts": {
                    "north": 1, "south": 1, "east": 1, "west": 0,
                },
                "queue_lengths": {"north": 1, "south": 1, "east": 1},
                "approaching_right_turn_counts": {
                    "north": 1, "south": 1, "east": 1,
                },
                "queued_right_turn_counts": {
                    "north": 1, "south": 1, "east": 1,
                },
            }
        )
        controller.update(0.01)

        self.assertEqual(controller.active_phase, "ns")
        self.assertEqual(controller.get_right_turn_state("north"), "green")
        self.assertEqual(controller.get_right_turn_state("south"), "green")
        self.assertEqual(controller.get_right_turn_state("east"), "red")
        self.assertEqual(controller.get_vehicle_state(north_right), "green")
        self.assertEqual(controller.get_vehicle_state(south_right), "green")

    def test_right_arrow_debounces_temporary_demand_loss(self):
        timing = self.config["traffic_lights"]
        timing["right_turn_demand_hold_s"] = 1.0
        timing["right_turn_min_green_s"] = 2.0
        right_demand = {"north": 1}
        controller = SixPhaseTrafficLightController(self.config)
        controller.set_phase_observation_provider(
            lambda: {
                "vehicle_counts": {"north": right_demand["north"]},
                "queue_lengths": {},
                "approaching_right_turn_counts": right_demand.copy(),
            }
        )

        controller.update(0.01)
        self.assertEqual(controller.get_right_turn_state("north"), "green")

        right_demand["north"] = 0
        controller.update(0.5)
        self.assertEqual(controller.get_right_turn_state("north"), "green")

        controller.update(1.6)
        self.assertEqual(controller.get_right_turn_state("north"), "off")

    def test_inactive_right_arrow_uses_permissive_main_green(self):
        controller = SixPhaseTrafficLightController(self.config)
        right_vehicle = SimpleNamespace(
            road_direction="north",
            is_turning_vehicle=True,
            turn_side="right",
            has_turned=False,
        )

        controller.update(0.01)

        self.assertEqual(controller.get_right_turn_state("north"), "off")
        self.assertEqual(controller.get_vehicle_state(right_vehicle), "green")

    def test_right_turn_demand_exposes_only_a_compatible_main_phase(self):
        controller = SixPhaseTrafficLightController(self.config)
        observation = {
            "vehicle_counts": {
                "north": 0, "south": 0, "east": 1, "west": 0,
            },
            "queue_lengths": {"east": 1},
            "approaching_right_turn_counts": {"east": 1},
            "queued_right_turn_counts": {"east": 1},
        }

        available = controller.get_available_phases(observation)

        self.assertEqual(available, ("ew",))

    def test_scene_guard_blocks_only_its_right_arrow(self):
        controller = SixPhaseTrafficLightController(self.config)
        north_right = SimpleNamespace(
            road_direction="north",
            is_turning_vehicle=True,
            turn_side="right",
            has_turned=False,
        )
        controller.set_phase_observation_provider(
            lambda: {
                "vehicle_counts": {"north": 1, "south": 1},
                "queue_lengths": {"north": 1, "south": 1},
                "approaching_right_turn_counts": {"north": 1, "south": 1},
                "queued_right_turn_counts": {"north": 1, "south": 1},
            }
        )
        controller.set_right_turn_activation_guard(
            lambda direction: direction != "north"
        )

        controller.update(0.01)

        self.assertEqual(controller.get_right_turn_state("north"), "red")
        self.assertEqual(controller.get_right_turn_state("south"), "green")
        self.assertEqual(controller.get_vehicle_state(north_right), "red")

    def test_right_arrow_safety_red_overrides_minimum_green(self):
        controller = SixPhaseTrafficLightController(self.config)
        controller.set_phase_observation_provider(
            lambda: {
                "vehicle_counts": {"north": 1},
                "queue_lengths": {"north": 1},
                "approaching_right_turn_counts": {"north": 1},
            }
        )
        controller.update(0.01)
        self.assertEqual(controller.get_right_turn_state("north"), "green")

        controller.set_right_turn_activation_guard(lambda direction: False)
        controller.update(0.01)

        self.assertEqual(controller.get_right_turn_state("north"), "red")

    def test_left_arrow_tracks_red_time_separately(self):
        controller = SixPhaseTrafficLightController(self.config)
        controller.set_phase_observation_provider(
            lambda: {
                "vehicle_counts": {
                    "north": 1, "south": 0, "east": 0, "west": 0,
                },
                "queue_lengths": {},
                "approaching_left_turn_counts": {"north": 1},
            }
        )

        controller.update(0.05)

        self.assertEqual(controller.get_state("north"), "green")
        self.assertEqual(controller.get_left_turn_state("north"), "red")
        self.assertEqual(controller.get_red_elapsed()["north"], 0.0)
        self.assertAlmostEqual(controller.get_left_red_elapsed()["north"], 0.05)

    def test_all_red_waits_for_committed_left_turn_to_clear(self):
        simulation = Simulation(self.config, random_seed=1)
        turning_vehicle = SimpleNamespace(turning=True, turn_side="left")
        simulation.vehicles = [turning_vehicle]

        self.assertFalse(simulation._can_activate_phase("ew"))

        turning_vehicle.turning = False
        self.assertTrue(simulation._can_activate_phase("ew"))

    def test_empty_approaches_do_not_force_max_green_switch(self):
        timing = self.config["traffic_lights"]
        timing["min_green_duration_s"] = 0.1
        timing["max_green_duration_s"] = 0.2
        controller = SixPhaseTrafficLightController(self.config)
        controller.set_phase_observation_provider(
            lambda: {
                "vehicle_counts": {
                    "north": 0, "south": 0, "east": 0, "west": 0,
                },
                "queue_lengths": {},
            }
        )
        controller.set_phase_selector(
            lambda active_phase, available_phases: "west_only"
        )

        controller.update(0.25)

        self.assertEqual(controller.phase_state, "green")
        self.assertEqual(controller.active_phase, "ns")

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

    def test_pre_intersection_wait_is_separated_by_entry_direction(self):
        metrics = Metrics(self.config)
        north_vehicle = SimpleNamespace(
            current_speed=0.0,
            last_deceleration_mps2=0.0,
            stopped=True,
            cleared_intersection=False,
            road_direction="north",
            turning=False,
        )
        south_vehicle = SimpleNamespace(
            current_speed=0.0,
            last_deceleration_mps2=0.0,
            stopped=True,
            cleared_intersection=True,
            road_direction="south",
            turning=False,
        )

        metrics.update([north_vehicle, south_vehicle], 2.0)
        summary = metrics.get_summary()

        by_direction = summary["avg_pre_intersection_wait_time_by_direction"]
        self.assertAlmostEqual(by_direction["north"], 2.0)
        self.assertAlmostEqual(by_direction["south"], 0.0)
        self.assertAlmostEqual(summary["avg_pre_intersection_wait_time"], 1.0)
        self.assertAlmostEqual(summary["max_avg_pre_intersection_wait_time"], 2.0)
        self.assertAlmostEqual(summary["pre_intersection_wait_time_imbalance"], 2.0)

    def test_control_metrics_track_wasted_green_switches_and_turn_delay(self):
        metrics = Metrics(self.config)
        controller = SixPhaseTrafficLightController(self.config)
        vehicle = SimpleNamespace(
            cleared_intersection=False,
            road_direction="east",
            current_speed=0.0,
            speed=100.0,
            is_turning_vehicle=True,
            turn_side="left",
            has_turned=False,
        )
        right_vehicle = SimpleNamespace(
            cleared_intersection=False,
            road_direction="east",
            current_speed=0.0,
            speed=100.0,
            is_turning_vehicle=True,
            turn_side="right",
            has_turned=False,
        )

        metrics.update_control(
            controller,
            [vehicle, right_vehicle],
            dt=2.0,
            intersection_stuck_vehicles=2,
        )
        controller.active_phase = "ew"
        metrics.update_control(controller, [vehicle, right_vehicle], dt=1.0)
        summary = metrics.get_summary()

        self.assertAlmostEqual(summary["empty_phase_time"], 3.0)
        self.assertAlmostEqual(summary["green_movement_utilization"], 0.0)
        self.assertAlmostEqual(
            summary["wasted_green_movement_fraction"],
            1.0,
        )
        self.assertAlmostEqual(summary["intersection_blocking_time"], 4.0)
        self.assertAlmostEqual(summary["left_turn_delay"], 3.0)
        self.assertAlmostEqual(summary["right_turn_delay"], 3.0)
        self.assertEqual(summary["phase_switches"], 1)
        self.assertEqual(summary["movement_set_changes"], 1)

    def test_normalized_turn_delay_applies_only_to_control_fitness(self):
        metrics = {
            "throughput_rate": 1.0,
            "avg_left_turn_delay": 2.0,
            "avg_right_turn_delay": 1.0,
        }
        base = calculate_fitness(metrics)
        six_phase = calculate_six_phase_fitness(
            metrics,
            six_phase_config={
                "avg_left_turn_delay_penalty": 20.0,
                "avg_right_turn_delay_penalty": 25.0,
            },
        )

        self.assertAlmostEqual(base, 10000.0)
        self.assertAlmostEqual(six_phase, 9935.0)

    def test_gridlock_receives_heavy_fitness_penalty(self):
        metrics = {
            "throughput_rate": 1.0,
            "gridlock_detected": 1,
            "gridlock_remaining_time_s": 20.0,
        }

        fitness = calculate_six_phase_fitness(
            metrics,
            six_phase_config={
                "gridlock_penalty": 100000.0,
                "gridlock_remaining_time_penalty": 1000.0,
            },
        )

        self.assertAlmostEqual(fitness, -110000.0)

    def test_dense_control_metrics_reduce_six_phase_fitness(self):
        metrics = {
            "throughput_rate": 0.0,
            "transition_clearance_fraction": 0.2,
            "wasted_green_movement_fraction": 0.3,
            "intersection_blocking_rate": 0.4,
            "avg_left_turn_delay": 5.0,
            "avg_right_turn_delay": 2.0,
        }

        fitness = calculate_six_phase_fitness(
            metrics,
            six_phase_config={
                "transition_clearance_fraction_penalty": 50.0,
                "wasted_green_movement_fraction_penalty": 25.0,
                "intersection_blocking_rate_penalty": 40.0,
                "avg_left_turn_delay_penalty": 15.0,
                "avg_right_turn_delay_penalty": 10.0,
            },
        )

        self.assertAlmostEqual(fitness, -128.5)

    def test_worst_approach_wait_reduces_six_phase_fitness(self):
        metrics = {
            "throughput": 0.0,
            "avg_wait_time": 0.0,
            "avg_active_wait_time": 0.0,
            "max_wait_time": 0.0,
            "queue_lengths": {},
            "max_avg_pre_intersection_wait_time": 4.0,
        }

        fitness = calculate_six_phase_fitness(
            metrics,
            six_phase_config={"worst_approach_wait_time_penalty": 3.0},
        )

        self.assertAlmostEqual(fitness, -12.0)

    def test_default_six_phase_fitness_avoids_overlapping_penalties(self):
        weights = self.config["six_phase_fitness"]

        self.assertGreater(
            weights["transition_clearance_fraction_penalty"],
            0.0,
        )
        self.assertGreater(
            weights["wasted_green_movement_fraction_penalty"],
            0.0,
        )
        self.assertGreater(weights["intersection_blocking_rate_penalty"], 0.0)
        self.assertGreater(weights["avg_left_turn_delay_penalty"], 0.0)
        self.assertGreater(weights["avg_right_turn_delay_penalty"], 0.0)
        self.assertGreater(weights["worst_approach_wait_time_penalty"], 0.0)

    def test_normal_turn_speed_does_not_create_turn_delay(self):
        metrics = Metrics(self.config)
        controller = SixPhaseTrafficLightController(self.config)
        vehicle = SimpleNamespace(
            cleared_intersection=False,
            road_direction="north",
            current_speed=20.0,
            speed=50.0,
            left_turn_speed=20.0,
            is_turning_vehicle=True,
            turn_side="left",
            has_turned=False,
        )

        metrics.update_control(controller, [vehicle], dt=2.0)

        self.assertAlmostEqual(metrics.get_summary()["left_turn_delay"], 0.0)

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

        occupancy, blocked = simulation.get_intersection_vehicle_counts(0.5)
        count = simulation.count_stuck_vehicles_in_intersection(0.5)

        self.assertEqual(occupancy, 2)
        self.assertEqual(blocked, 1)
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
                    "phase_switches",
                    "empty_phase_time",
                    "intersection_blocking_time",
                    "left_turn_delay",
                    "paired_phase_time",
                    "single_phase_time",
                    "gridlock_detected",
                    "max_intersection_stuck_vehicles",
                    "evaluation_elapsed_s",
                )
            },
            "random_seed": 1,
            "traffic_profile": "balanced_no_turns",
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
        self.assertEqual(result["evaluated_scenario_count"], 1)
        self.assertEqual(
            result["requested_scenario_count"],
            len(self.config["six_phase_training"]["traffic_profiles"]) * 3,
        )
        self.assertEqual(
            result["skipped_scenario_count"],
            result["requested_scenario_count"] - 1,
        )
        self.assertAlmostEqual(result["mean_fitness"], -100000.0)
        self.assertTrue(result["candidate_rejected"])

    def test_exact_scenario_pairs_bypass_cartesian_profiles_and_seeds(self):
        requested_pairs = (
            ({"name": "turning", "left_turn_chance": 0.4}, 17),
            ({"name": "quiet", "arrival_rates_per_s": {}}, 4),
        )
        calls = []

        def evaluate_exact(*args, **kwargs):
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

        policy = SimpleNamespace(select_phase=lambda *args: "ns")
        with patch(
            "simulation.evaluation.evaluate_six_phase_policy",
            side_effect=evaluate_exact,
        ):
            result = evaluate_six_phase_policy_across_seeds(
                self.config,
                policy,
                seeds=(999,),
                traffic_profiles=({"name": "ignored"},),
                scenario_pairs=requested_pairs,
            )

        self.assertEqual(calls, [("turning", 17), ("quiet", 4)])
        self.assertEqual(result["requested_scenarios"], tuple(calls))
        self.assertEqual(result["evaluated_scenarios"], tuple(calls))
        self.assertEqual(result["skipped_scenarios"], ())
        self.assertEqual(result["requested_scenario_count"], 2)
        self.assertEqual(result["evaluated_scenario_count"], 2)
        self.assertAlmostEqual(result["mean_fitness"], 10.5)

    def test_gridlock_imputes_failure_fitness_for_every_skipped_scenario(self):
        scenario_pairs = tuple(
            ({"name": f"scenario_{index}"}, index)
            for index in range(4)
        )
        completed = {
            "fitness": 100.0,
            "metrics": {},
            "random_seed": 0,
            "traffic_profile": "scenario_0",
            "terminated_early": False,
            "termination_reason": None,
        }
        gridlocked = {
            "fitness": -100000.0,
            "metrics": {"gridlock_detected": 1.0},
            "random_seed": 1,
            "traffic_profile": "scenario_1",
            "terminated_early": True,
            "termination_reason": "intersection_gridlock",
        }
        policy = SimpleNamespace(select_phase=lambda *args: "ns")

        with patch(
            "simulation.evaluation.evaluate_six_phase_policy",
            side_effect=(completed, gridlocked),
        ) as evaluate:
            result = evaluate_six_phase_policy_across_seeds(
                self.config,
                policy,
                scenario_pairs=scenario_pairs,
            )

        self.assertEqual(evaluate.call_count, 2)
        self.assertEqual(result["evaluated_scenario_count"], 2)
        self.assertEqual(result["skipped_scenario_count"], 2)
        self.assertEqual(
            result["skipped_scenarios"],
            (("scenario_2", 2), ("scenario_3", 3)),
        )
        self.assertAlmostEqual(
            result["mean_fitness"],
            (100.0 - 3 * 100000.0) / 4,
        )

    def test_gridlock_does_not_skip_scenarios_when_abort_is_disabled(self):
        config = {
            **self.config,
            "six_phase_fitness": {
                **self.config.get("six_phase_fitness", {}),
                "abort_remaining_seeds_on_gridlock": False,
            },
        }
        scenario_pairs = tuple(
            ({"name": f"scenario_{index}"}, index)
            for index in range(3)
        )

        def gridlock_result(*args, **kwargs):
            profile = kwargs["traffic_profile"]
            seed = kwargs["random_seed"]
            return {
                "fitness": -100000.0,
                "metrics": {"gridlock_detected": 1.0},
                "random_seed": seed,
                "traffic_profile": profile["name"],
                "terminated_early": True,
                "termination_reason": "intersection_gridlock",
            }

        policy = SimpleNamespace(select_phase=lambda *args: "ns")
        with patch(
            "simulation.evaluation.evaluate_six_phase_policy",
            side_effect=gridlock_result,
        ) as evaluate:
            result = evaluate_six_phase_policy_across_seeds(
                config,
                policy,
                scenario_pairs=scenario_pairs,
            )

        self.assertEqual(evaluate.call_count, 3)
        self.assertEqual(result["evaluated_scenario_count"], 3)
        self.assertEqual(result["skipped_scenario_count"], 0)
        self.assertTrue(result["candidate_rejected"])


if __name__ == "__main__":
    unittest.main()
