import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from config import CONFIG, build_runtime_config
from simulation import Simulation
from simulation.fixed_time import (
    FixedTimeMovementPlan,
    FixedTimeMovementTrafficLightController,
    FixedTimeStage,
    load_fixed_time_plan,
)


class FixedTimePlanTests(unittest.TestCase):
    @staticmethod
    def data():
        return {
            "format_version": 1,
            "policy_type": "fixed_time_movement",
            "name": "test_plan",
            "stages": [
                {
                    "name": "north_south",
                    "duration_s": 2.0,
                    "movements": [
                        "south_right",
                        "north_through",
                        "south_through",
                        "north_right",
                    ],
                },
                {
                    "name": "east_west",
                    "duration_s": 3.0,
                    "movements": ["east_through", "west_through"],
                },
            ],
            "metadata": {"calibration": {"seed": 7}},
        }

    def test_round_trip_is_versioned_canonical_and_independent(self):
        plan = FixedTimeMovementPlan.from_dict(self.data())
        serialized = plan.to_dict()

        self.assertEqual(serialized["format_version"], 1)
        self.assertEqual(serialized["policy_type"], "fixed_time_movement")
        self.assertEqual(
            serialized["stages"][0]["movements"],
            [
                "north_through",
                "south_through",
                "north_right",
                "south_right",
            ],
        )
        serialized["metadata"]["calibration"]["seed"] = 99
        self.assertEqual(plan.metadata["calibration"]["seed"], 7)
        self.assertEqual(
            FixedTimeMovementPlan.from_dict(plan.to_dict()),
            plan,
        )

    def test_loader_reads_validated_json(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "fixed.json"
            path.write_text(json.dumps(self.data()), encoding="utf-8")
            plan = load_fixed_time_plan(path)

        self.assertEqual(plan.name, "test_plan")
        self.assertEqual(len(plan.stages), 2)

    def test_rejects_unknown_conflicting_and_invalid_stage_values(self):
        invalid = self.data()
        invalid["stages"][0]["movements"] = ["north_flying"]
        with self.assertRaisesRegex(ValueError, "unknown movements"):
            FixedTimeMovementPlan.from_dict(invalid)

        invalid = self.data()
        invalid["stages"][0]["movements"] = [
            "north_through",
            "east_through",
        ]
        with self.assertRaisesRegex(ValueError, "conflicting movements"):
            FixedTimeMovementPlan.from_dict(invalid)

        invalid = self.data()
        invalid["stages"][0]["duration_s"] = 0.0
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            FixedTimeMovementPlan.from_dict(invalid)

        invalid = self.data()
        invalid["format_version"] = 2
        with self.assertRaisesRegex(ValueError, "format_version"):
            FixedTimeMovementPlan.from_dict(invalid)

    def test_direct_dataclass_construction_is_also_validated(self):
        with self.assertRaisesRegex(ValueError, "conflicting movements"):
            FixedTimeStage(
                "unsafe",
                5.0,
                ("north_through", "east_through"),
            )


class FixedTimeControllerTests(unittest.TestCase):
    def setUp(self):
        self.config = build_runtime_config(CONFIG)
        self.config["traffic_lights"]["yellow_duration_s"] = 0.5
        self.config["traffic_lights"]["all_red_clearance_duration_s"] = 0.25
        self.config["traffic_lights"]["min_green_duration_s"] = 10.0
        self.config["traffic_lights"]["max_green_duration_s"] = 30.0
        self.config["movement_controller"]["empty_green_gap_out_s"] = 0.01
        self.plan = FixedTimeMovementPlan(
            name="cycle",
            stages=(
                FixedTimeStage(
                    "north_south",
                    2.0,
                    (
                        "north_through",
                        "south_through",
                        "north_right",
                        "south_right",
                    ),
                ),
                FixedTimeStage(
                    "east_west",
                    3.0,
                    ("east_through", "west_through"),
                ),
            ),
        )

    def activate_first_stage(self, controller):
        controller.update(0.24)
        self.assertEqual(controller.phase_state, "all_red")
        controller.update(0.01)
        self.assertEqual(controller.phase_state, "green")

    def test_starts_all_red_and_cycles_full_duration_with_explicit_rights(self):
        controller = FixedTimeMovementTrafficLightController(
            self.config,
            self.plan,
        )
        self.assertEqual(controller.phase_state, "all_red")
        self.assertEqual(controller.active_phase, "none")
        self.assertTrue(
            all(state == "red" for state in controller.states.values())
        )
        self.assertIsNone(controller.last_movement_scores)

        self.activate_first_stage(controller)
        self.assertEqual(controller.active_stage_name, "north_south")
        self.assertEqual(
            controller.active_movements,
            frozenset(("north_through", "south_through")),
        )
        self.assertEqual(controller.get_right_turn_state("north"), "green")
        self.assertEqual(controller.get_right_turn_state("south"), "green")
        self.assertEqual(controller.get_right_turn_state("east"), "red")
        self.assertEqual(
            controller.get_active_policy_movements(),
            frozenset(self.plan.stages[0].movements),
        )

        # No traffic exists, max/min green disagree with the plan, and the
        # configured neural gap-out is tiny. The fixed stage still runs 2 s.
        controller.update(1.99)
        self.assertEqual(controller.phase_state, "green")
        controller.update(0.01)
        self.assertEqual(controller.phase_state, "yellow")
        self.assertEqual(controller.get_right_turn_state("north"), "red")
        self.assertEqual(controller.get_state("north"), "yellow")

        controller.update(0.5)
        self.assertEqual(controller.phase_state, "all_red")
        controller.update(0.25)
        self.assertEqual(controller.phase_state, "green")
        self.assertEqual(controller.active_stage_name, "east_west")
        self.assertEqual(controller.get_state("east"), "green")
        self.assertEqual(controller.current_green_duration, 3.0)
        self.assertEqual(controller.empty_green_gap_out_count, 0)

    def test_main_stage_waits_for_scene_guard_after_clearance(self):
        controller = FixedTimeMovementTrafficLightController(
            self.config,
            self.plan,
        )
        allowed = False
        controller.set_movement_activation_guard(lambda movements: allowed)

        controller.update(1.0)
        self.assertEqual(controller.phase_state, "all_red")
        self.assertGreaterEqual(controller.timer, 1.0)
        allowed = True
        controller.update(0.01)
        self.assertEqual(controller.phase_state, "green")
        self.assertEqual(controller.timer, 0.0)

    def test_unsafe_explicit_right_is_suppressed_without_holding_main_stage(self):
        controller = FixedTimeMovementTrafficLightController(
            self.config,
            self.plan,
        )
        north_right_allowed = False
        controller.set_right_turn_activation_guard(
            lambda direction: direction != "north" or north_right_allowed
        )

        controller.update(0.25)
        self.assertEqual(controller.phase_state, "green")
        self.assertEqual(controller.get_state("north"), "green")
        self.assertEqual(controller.get_right_turn_state("north"), "red")
        self.assertEqual(controller.get_right_turn_state("south"), "green")

        north_right_allowed = True
        controller.update(0.01)
        self.assertEqual(controller.get_right_turn_state("north"), "green")

    def test_simulation_selects_fixed_controller_and_rejects_adaptive_mix(self):
        simulation = Simulation(self.config, fixed_time_plan=self.plan)
        self.assertIsInstance(
            simulation.light_controller,
            FixedTimeMovementTrafficLightController,
        )
        self.assertIsNotNone(
            simulation.light_controller.phase_observation_provider
        )
        self.assertIsNotNone(
            simulation.light_controller.movement_activation_guard
        )
        self.assertIsNotNone(
            simulation.light_controller.right_turn_activation_guard
        )

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            Simulation(
                self.config,
                fixed_time_plan=self.plan,
                movement_score_provider=lambda observation: {},
            )


if __name__ == "__main__":
    unittest.main()
