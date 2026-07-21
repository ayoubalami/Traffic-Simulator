import unittest

from config import CONFIG, build_runtime_config
from simulation.traffic_light import MovementTrafficLightController


class MovementPedestrianControllerTests(unittest.TestCase):
    DIRECTIONS = ("north", "south", "east", "west")

    def setUp(self):
        self.config = build_runtime_config(CONFIG)
        self.config.setdefault("road_users", {})[
            "pedestrians_enabled"
        ] = True
        # These tests arrange established green/pedestrian states directly;
        # policy-selected startup is covered separately by movement tests.
        self.config["movement_controller"][
            "policy_selected_initial_phase"
        ] = False
        timing = self.config["traffic_lights"]
        timing["min_green_duration_s"] = 0.1
        timing["green_extension_check_interval_s"] = 0.1
        timing["yellow_duration_s"] = 0.1
        timing["all_red_clearance_duration_s"] = 0.05
        pedestrian = self.config["pedestrian_signals"]
        pedestrian["min_walk_duration_s"] = 0.2
        pedestrian["max_walk_duration_s"] = 0.5
        pedestrian["max_red_duration_s"] = 1.0
        pedestrian["clearance_duration_s"] = 0.1

    def observation(self, *, vehicles=None, waiting=None):
        zeros = {direction: 0 for direction in self.DIRECTIONS}
        vehicles = {**zeros, **(vehicles or {})}
        waiting = {**zeros, **(waiting or {})}
        return {
            "vehicle_counts": vehicles,
            "queue_lengths": vehicles.copy(),
            "approaching_left_turn_counts": zeros.copy(),
            "queued_left_turn_counts": zeros.copy(),
            "approaching_right_turn_counts": zeros.copy(),
            "queued_right_turn_counts": zeros.copy(),
            "waiting_pedestrian_counts": waiting,
        }

    @staticmethod
    def combined_scores(*, vehicle=(), walks=()):
        scores = {
            movement: 0.0
            for movement in MovementTrafficLightController.MOVEMENTS
        }
        scores.update(
            {
                output: 0.0
                for output in MovementTrafficLightController.PEDESTRIAN_OUTPUTS
            }
        )
        scores.update({name: 0.9 for name in vehicle})
        scores.update({name: 0.9 for name in walks})
        return scores

    def test_output_schema_is_separate_from_vehicle_movements(self):
        self.assertEqual(
            MovementTrafficLightController.PEDESTRIAN_OUTPUTS,
            (
                "north_walk",
                "south_walk",
                "east_walk",
                "west_walk",
            ),
        )
        self.assertNotIn(
            "north_walk",
            MovementTrafficLightController.MOVEMENTS,
        )

    def test_through_movement_conflicts_with_entry_and_exit_crosswalks(self):
        controller = MovementTrafficLightController(self.config)

        self.assertEqual(
            controller._movement_crossings("north_through"),
            {"north", "south"},
        )
        self.assertEqual(
            controller._movement_crossings("east_through"),
            {"east", "west"},
        )
        self.assertEqual(
            controller._movement_crossings("north_left"),
            {"north", "east"},
        )
        self.assertEqual(
            controller._movement_crossings("north_right"),
            {"north", "west"},
        )

    def test_legacy_twelve_output_provider_keeps_automatic_walk_behavior(self):
        controller = MovementTrafficLightController(self.config)
        observation = self.observation(vehicles={"north": 1})
        controller.set_phase_observation_provider(lambda: observation)
        # Remove the WALK keys to emulate every existing saved policy.
        controller.set_movement_score_provider(
            lambda current: {
                movement: 0.9 if movement == "north_through" else 0.0
                for movement in controller.MOVEMENTS
            }
        )

        controller.update(0.01)

        self.assertFalse(controller._pedestrian_policy_enabled)
        self.assertEqual(controller.get_pedestrian_state("east"), "green")
        self.assertEqual(controller.get_active_pedestrian_outputs(), frozenset())

    def test_walk_request_clears_conflicting_through_before_activation(self):
        controller = MovementTrafficLightController(self.config)
        observation = self.observation(
            vehicles={"north": 1},
            waiting={"south": 1},
        )
        crosswalk_clear = {direction: True for direction in self.DIRECTIONS}
        crosswalk_clear["south"] = False
        controller.set_phase_observation_provider(lambda: observation)
        controller.set_movement_score_provider(
            lambda current: self.combined_scores(
                vehicle=("north_through",),
                walks=("south_walk",),
            )
        )
        controller.set_crosswalk_vehicle_occupancy_guard(
            lambda crossing: crosswalk_clear[crossing]
        )

        controller.update(0.11)

        self.assertEqual(controller.phase_state, "yellow")
        self.assertEqual(controller.pending_movements, frozenset())
        self.assertEqual(controller.get_pedestrian_state("south"), "red")
        self.assertEqual(
            controller.pending_pedestrian_outputs,
            frozenset(("south_walk",)),
        )

        controller.update(0.11)
        self.assertEqual(controller.phase_state, "all_red")
        controller.update(0.06)
        self.assertEqual(controller.phase_state, "green")
        self.assertEqual(controller.active_movements, frozenset())
        self.assertEqual(controller.get_pedestrian_state("south"), "red")

        crosswalk_clear["south"] = True
        controller.update(0.01)

        self.assertEqual(controller.get_pedestrian_state("south"), "green")
        self.assertEqual(
            controller.get_active_pedestrian_outputs(),
            frozenset(("south_walk",)),
        )
        self.assertEqual(
            controller.get_active_pedestrian_walks(),
            frozenset(("south",)),
        )

    def test_dedicated_provider_can_request_multiple_safe_walks(self):
        controller = MovementTrafficLightController(self.config)
        controller.active_movements = frozenset()
        controller.active_phase = "none"
        controller.states = {
            direction: "red" for direction in self.DIRECTIONS
        }
        controller.right_turn_states = {
            direction: "red" for direction in self.DIRECTIONS
        }
        observation = self.observation(
            waiting={"north": 1, "east": 2},
        )
        controller.set_phase_observation_provider(lambda: observation)
        controller.set_movement_score_provider(
            lambda current: {
                movement: 0.0 for movement in controller.MOVEMENTS
            }
        )
        controller.set_pedestrian_score_provider(
            lambda current: {
                output: 0.9
                if output in ("north_walk", "east_walk")
                else 0.0
                for output in controller.PEDESTRIAN_OUTPUTS
            }
        )

        controller.update(0.01)

        self.assertEqual(
            controller.get_active_pedestrian_outputs(),
            frozenset(("north_walk", "east_walk")),
        )

    def test_walk_waits_for_all_red_clearance_and_blocks_pending_conflict(self):
        controller = MovementTrafficLightController(self.config)
        observation = self.observation(
            vehicles={"north": 1},
            waiting={"south": 1},
        )
        controller.set_phase_observation_provider(lambda: observation)
        controller.set_movement_score_provider(
            lambda current: self.combined_scores(
                vehicle=("north_through",),
                walks=("south_walk",),
            )
        )
        controller._refresh_movement_scores(observation)
        controller.phase_state = "all_red"
        controller.timer = 0.0
        controller.active_movements = frozenset()
        controller.pending_movements = frozenset(("north_through",))
        controller.states = {
            direction: "red" for direction in self.DIRECTIONS
        }
        controller.right_turn_states = {
            direction: "red" for direction in self.DIRECTIONS
        }

        controller.update(0.02)

        self.assertEqual(controller.get_pedestrian_state("south"), "red")
        self.assertEqual(controller.phase_state, "all_red")

        controller.update(0.04)

        self.assertEqual(controller.get_pedestrian_state("south"), "green")
        self.assertEqual(controller.phase_state, "all_red")
        self.assertEqual(
            controller.pending_movements,
            frozenset(("north_through",)),
        )

    def test_walk_minimum_and_clearance_survive_score_drop(self):
        controller = MovementTrafficLightController(self.config)
        controller.active_movements = frozenset()
        controller.active_phase = "none"
        controller.states = {
            direction: "red" for direction in self.DIRECTIONS
        }
        controller.right_turn_states = {
            direction: "red" for direction in self.DIRECTIONS
        }
        observation = self.observation(waiting={"east": 1})
        walk_requested = {"value": True}
        controller.set_phase_observation_provider(lambda: observation)
        controller.set_movement_score_provider(
            lambda current: self.combined_scores(
                walks=("east_walk",) if walk_requested["value"] else (),
            )
        )

        controller.update(0.01)
        self.assertEqual(controller.get_pedestrian_state("east"), "green")

        walk_requested["value"] = False
        controller.next_score_update = 0.0
        controller.update(0.10)
        self.assertEqual(controller.get_pedestrian_state("east"), "green")

        controller.next_score_update = 0.0
        controller.update(0.11)
        self.assertEqual(controller.get_pedestrian_state("east"), "red")
        self.assertIn(
            "east",
            controller._pedestrian_vehicle_blocking_crossings(),
        )

        controller.update(0.11)
        self.assertNotIn(
            "east",
            controller._pedestrian_vehicle_blocking_crossings(),
        )

    def test_continuous_request_closes_at_maximum_walk_duration(self):
        self.config["pedestrian_signals"]["min_walk_duration_s"] = 0.1
        self.config["pedestrian_signals"]["max_walk_duration_s"] = 0.2
        controller = MovementTrafficLightController(self.config)
        controller.active_movements = frozenset()
        controller.active_phase = "none"
        controller.states = {
            direction: "red" for direction in self.DIRECTIONS
        }
        controller.right_turn_states = {
            direction: "red" for direction in self.DIRECTIONS
        }
        observation = self.observation(waiting={"north": 1})
        controller.set_phase_observation_provider(lambda: observation)
        controller.set_movement_score_provider(
            lambda current: self.combined_scores(walks=("north_walk",))
        )

        controller.update(0.01)
        self.assertEqual(controller.get_pedestrian_state("north"), "green")
        controller.update(0.21)

        self.assertEqual(controller.get_pedestrian_state("north"), "red")
        self.assertGreater(
            controller.pedestrian_clearance_remaining["north"],
            0.0,
        )

    def test_max_red_fairness_requests_walk_even_with_low_score(self):
        self.config["pedestrian_signals"]["min_walk_duration_s"] = 0.1
        self.config["pedestrian_signals"]["max_red_duration_s"] = 0.2
        controller = MovementTrafficLightController(self.config)
        controller.active_movements = frozenset()
        controller.active_phase = "none"
        controller.states = {
            direction: "red" for direction in self.DIRECTIONS
        }
        controller.right_turn_states = {
            direction: "red" for direction in self.DIRECTIONS
        }
        observation = self.observation(waiting={"west": 1})
        controller.set_phase_observation_provider(lambda: observation)
        controller.set_movement_score_provider(
            lambda current: self.combined_scores()
        )

        controller.update(0.10)
        self.assertEqual(controller.get_pedestrian_state("west"), "red")
        controller.update(0.11)

        self.assertEqual(controller.get_pedestrian_state("west"), "green")
        self.assertNotIn(
            "west_walk",
            controller.last_raw_requested_pedestrian_outputs,
        )
        self.assertIn(
            "west_walk",
            controller.last_decoded_pedestrian_outputs,
        )
        self.assertEqual(controller.get_pedestrian_red_elapsed()["west"], 0.0)

    def test_requested_walk_forces_conflicting_permissive_right_red(self):
        controller = MovementTrafficLightController(self.config)
        observation = self.observation(waiting={"west": 1})
        controller.set_phase_observation_provider(lambda: observation)
        controller.set_movement_score_provider(
            lambda current: self.combined_scores(walks=("west_walk",))
        )

        controller.update(0.01)

        self.assertEqual(controller.get_right_turn_state("north"), "red")
        self.assertEqual(
            controller.get_right_turn_permission_state("north"),
            "red",
        )


if __name__ == "__main__":
    unittest.main()
