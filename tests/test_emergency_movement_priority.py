import unittest

from config import CONFIG, build_runtime_config
from simulation.movement_neuroevolution import MOVEMENT_NAMES
from simulation.traffic_light import MovementTrafficLightController


class EmergencyMovementPriorityTests(unittest.TestCase):
    DIRECTIONS = ("north", "south", "east", "west")

    def config(self, *, startup=False):
        config = build_runtime_config(CONFIG)
        config["movement_controller"]["policy_selected_initial_phase"] = startup
        config["movement_controller"]["switch_hysteresis"] = 0.95
        return config

    def observation(
        self,
        *,
        vehicles=None,
        left=None,
        right=None,
        emergencies=None,
    ):
        zeros = {direction: 0 for direction in self.DIRECTIONS}
        vehicles = {**zeros, **(vehicles or {})}
        left = {**zeros, **(left or {})}
        right = {**zeros, **(right or {})}
        return {
            "vehicle_counts": vehicles,
            "queue_lengths": vehicles.copy(),
            "average_speed_ratios": zeros.copy(),
            "emergency_counts": {**zeros, **(emergencies or {})},
            "approaching_left_turn_counts": left,
            "queued_left_turn_counts": left.copy(),
            "approaching_right_turn_counts": right,
            "queued_right_turn_counts": right.copy(),
            "red_elapsed_s": zeros.copy(),
            "left_red_elapsed_s": zeros.copy(),
            "right_red_elapsed_s": zeros.copy(),
            "active_movements": ("north_through", "south_through"),
            "green_elapsed_s": 0.0,
        }

    @staticmethod
    def scores(**priorities):
        return {
            movement: float(priorities.get(movement, 0.0))
            for movement in MOVEMENT_NAMES
        }

    def test_emergency_outranks_neural_score_and_hysteresis(self):
        controller = MovementTrafficLightController(self.config())
        observation = self.observation(
            vehicles={"north": 8, "east": 1},
            emergencies={"east": 1},
        )
        scores = self.scores(north_through=0.99, east_through=0.01)

        decoded = controller.decode_scores(scores, observation)

        self.assertEqual(decoded, frozenset(("east_through",)))
        self.assertIn(
            "east_through",
            controller.last_emergency_demanded_movements,
        )

    def test_explicit_emergency_turn_does_not_prioritize_other_same_side_flow(self):
        controller = MovementTrafficLightController(self.config())
        observation = self.observation(
            vehicles={"north": 5, "east": 1},
            left={"east": 1},
            emergencies={"east": 1},
        )
        observation["emergency_movement_counts"] = {
            movement: int(movement == "east_left")
            for movement in MOVEMENT_NAMES
        }

        decoded = controller.decode_scores(
            self.scores(north_through=0.99, east_left=0.01),
            observation,
        )

        self.assertIn("east_left", decoded)
        self.assertNotIn("north_through", decoded)
        self.assertEqual(
            controller.last_emergency_demanded_movements,
            frozenset(("east_left",)),
        )

    def test_conflicting_emergencies_use_longest_red_wait(self):
        controller = MovementTrafficLightController(self.config())
        observation = self.observation(
            vehicles={"north": 1, "east": 1},
            emergencies={"north": 1, "east": 1},
        )
        controller.movement_red_elapsed["north_through"] = 4.0
        controller.movement_red_elapsed["east_through"] = 12.0
        scores = self.scores(north_through=0.99, east_through=0.01)

        decoded = controller.decode_scores(scores, observation)

        self.assertEqual(decoded, frozenset(("east_through",)))
        self.assertTrue(controller.is_conflict_free(decoded))

    def test_emergency_preempts_minimum_green_but_keeps_clearance_and_guard(self):
        config = self.config()
        timing = config["traffic_lights"]
        timing["min_green_duration_s"] = 20.0
        timing["max_green_duration_s"] = 30.0
        timing["yellow_duration_s"] = 0.2
        timing["all_red_clearance_duration_s"] = 0.1
        controller = MovementTrafficLightController(config)
        observation = self.observation(
            vehicles={"north": 8, "east": 1},
            emergencies={"east": 1},
        )
        controller.set_phase_observation_provider(lambda: observation)
        controller.set_movement_score_provider(
            lambda current: self.scores(
                north_through=0.99,
                east_through=0.01,
            )
        )
        activation_allowed = {"value": False}
        controller.set_phase_activation_guard(
            lambda phase: activation_allowed["value"]
        )

        controller.update(0.01)

        self.assertEqual(controller.phase_state, "yellow")
        self.assertEqual(
            controller.pending_movements,
            frozenset(("east_through",)),
        )
        self.assertEqual(controller.states["north"], "yellow")
        self.assertEqual(controller.states["east"], "red")

        controller.update(0.21)
        self.assertEqual(controller.phase_state, "all_red")
        self.assertTrue(
            all(state == "red" for state in controller.states.values())
        )

        controller.update(0.11)
        self.assertEqual(controller.phase_state, "all_red")

        activation_allowed["value"] = True
        controller.update(0.01)
        self.assertEqual(controller.phase_state, "green")
        self.assertEqual(
            controller.active_movements,
            frozenset(("east_through",)),
        )

    def test_conflicting_emergencies_receive_stable_minimum_green(self):
        config = self.config()
        timing = config["traffic_lights"]
        timing["min_green_duration_s"] = 0.20
        timing["max_green_duration_s"] = 2.0
        timing["yellow_duration_s"] = 0.05
        timing["all_red_clearance_duration_s"] = 0.05
        config["movement_controller"][
            "emergency_min_green_duration_s"
        ] = 0.20
        controller = MovementTrafficLightController(config)
        observation = self.observation(
            vehicles={"north": 1, "east": 1},
            emergencies={"north": 1, "east": 1},
        )
        observation["emergency_movement_counts"] = {
            movement: int(
                movement in ("north_through", "east_through")
            )
            for movement in MOVEMENT_NAMES
        }
        controller.set_phase_observation_provider(lambda: observation)
        controller.set_movement_score_provider(
            lambda current: self.scores(
                north_through=0.99,
                east_through=0.01,
            )
        )

        # North already serves one emergency. East cannot interrupt it on the
        # first inference tick despite being another emergency demand.
        controller.update(0.01)
        self.assertEqual(controller.phase_state, "green")
        self.assertEqual(
            controller.active_movements,
            frozenset(("north_through", "south_through")),
        )
        controller.update(0.18)
        self.assertEqual(controller.phase_state, "green")

        # Once the emergency minimum expires, the older red emergency becomes
        # eligible through the normal yellow/all-red safety sequence.
        controller.update(0.02)
        self.assertEqual(controller.phase_state, "yellow")
        self.assertEqual(
            controller.pending_movements,
            frozenset(("east_through",)),
        )
        controller.update(0.11)
        self.assertEqual(controller.phase_state, "all_red")
        controller.update(0.06)
        self.assertEqual(controller.phase_state, "green")
        self.assertEqual(
            controller.active_movements,
            frozenset(("east_through",)),
        )

        # The persistent north emergency cannot take the new green back on
        # the next tick; east receives the same protected service interval.
        controller.update(0.01)
        self.assertEqual(controller.phase_state, "green")
        self.assertEqual(
            controller.active_movements,
            frozenset(("east_through",)),
        )
        controller.update(0.20)
        self.assertEqual(controller.phase_state, "yellow")
        self.assertEqual(
            controller.pending_movements,
            frozenset(("north_through",)),
        )

    def test_emergency_lock_releases_when_served_approach_clears(self):
        config = self.config()
        config["movement_controller"][
            "emergency_min_green_duration_s"
        ] = 20.0
        controller = MovementTrafficLightController(config)
        observation = self.observation(
            vehicles={"north": 1, "east": 1},
            emergencies={"north": 1, "east": 1},
        )
        controller.set_phase_observation_provider(lambda: observation)
        controller.set_movement_score_provider(
            lambda current: self.scores(
                north_through=0.99,
                east_through=0.01,
            )
        )

        controller.update(0.01)
        self.assertTrue(controller._emergency_service_is_locked())

        # This uses the legacy approach-only emergency observation. As soon as
        # the served north emergency clears, east can preempt without waiting
        # for the twenty-second lock timeout.
        observation["emergency_counts"]["north"] = 0
        controller.update(0.01)

        self.assertEqual(controller.phase_state, "yellow")
        self.assertEqual(
            controller.pending_movements,
            frozenset(("east_through",)),
        )

    def test_emergency_lock_defaults_to_normal_minimum_green(self):
        config = self.config()
        config["movement_controller"].pop(
            "emergency_min_green_duration_s",
            None,
        )
        config["traffic_lights"]["min_green_duration_s"] = 0.37

        controller = MovementTrafficLightController(config)

        self.assertAlmostEqual(
            controller.emergency_min_green_duration,
            0.37,
        )

    def test_max_green_bounds_a_stalled_emergency_service(self):
        config = self.config()
        timing = config["traffic_lights"]
        timing["min_green_duration_s"] = 0.1
        timing["max_green_duration_s"] = 0.3
        timing["green_extension_check_interval_s"] = 0.1
        config["movement_controller"][
            "emergency_min_green_duration_s"
        ] = 0.1
        controller = MovementTrafficLightController(config)
        observation = self.observation(
            vehicles={"north": 1, "east": 5},
            emergencies={"north": 1},
        )
        observation["emergency_movement_counts"] = {
            movement: int(movement == "north_through")
            for movement in MOVEMENT_NAMES
        }
        controller.set_phase_observation_provider(lambda: observation)
        controller.set_movement_score_provider(
            lambda current: self.scores(
                north_through=0.99,
                east_through=0.01,
            )
        )

        controller.update(0.31)

        self.assertEqual(controller.phase_state, "yellow")
        self.assertEqual(
            controller.pending_movements,
            frozenset(("east_through",)),
        )

    def test_movement_guard_can_reject_emergency_candidate(self):
        controller = MovementTrafficLightController(self.config())
        controller.set_movement_activation_guard(
            lambda movements: "east_through" not in movements
        )
        observation = self.observation(
            vehicles={"north": 1, "east": 1},
            emergencies={"east": 1},
        )

        decoded = controller.decode_scores(
            self.scores(north_through=0.1, east_through=0.99),
            observation,
        )

        self.assertNotIn("east_through", decoded)
        self.assertTrue(controller.is_conflict_free(decoded))

    def test_right_only_emergency_can_start_with_zero_network_score(self):
        config = self.config(startup=True)
        config["traffic_lights"]["all_red_clearance_duration_s"] = 0.05
        controller = MovementTrafficLightController(config)
        observation = self.observation(
            vehicles={"east": 1},
            right={"east": 1},
            emergencies={"east": 1},
        )
        observation["active_movements"] = ()
        controller.set_phase_observation_provider(lambda: observation)
        controller.set_movement_score_provider(
            lambda current: self.scores()
        )

        controller.update(0.01)

        self.assertEqual(controller.phase_state, "all_red")
        self.assertEqual(controller.pending_movements, frozenset())
        controller.update(0.06)

        self.assertEqual(controller.phase_state, "green")
        self.assertEqual(controller.active_movements, frozenset())
        self.assertEqual(controller.get_right_turn_state("east"), "green")


if __name__ == "__main__":
    unittest.main()
