import unittest

from config import CONFIG, build_runtime_config
from simulation import MovementPolicy, MovementPolicyEvolution
from simulation.movement_neuroevolution import (
    MOVEMENT_INPUT_FEATURE_NAMES,
    MOVEMENT_NAMES,
    POLICY_OUTPUT_NAMES,
)
from simulation.traffic_light import MovementTrafficLightController


class MovementPolicyTests(unittest.TestCase):
    def setUp(self):
        self.config = build_runtime_config(CONFIG)
        # Most tests below exercise steady-state decoding or transitions and
        # explicitly rely on the legacy active North/South setup. Startup has
        # dedicated coverage using the production default in the tests below.
        self.config["movement_controller"][
            "policy_selected_initial_phase"
        ] = False
        self.config["movement_controller"]["switch_hysteresis"] = 0.0

    @staticmethod
    def startup_config():
        config = build_runtime_config(CONFIG)
        config["movement_controller"][
            "policy_selected_initial_phase"
        ] = True
        config["movement_controller"]["switch_hysteresis"] = 0.0
        return config

    def observation(self, *, vehicles=None, left=None, right=None):
        directions = ("north", "south", "east", "west")
        zeros = {direction: 0 for direction in directions}
        vehicles = {**zeros, **(vehicles or {})}
        left = {**zeros, **(left or {})}
        right = {**zeros, **(right or {})}
        return {
            "vehicle_counts": vehicles,
            "queue_lengths": vehicles.copy(),
            "average_speed_ratios": zeros.copy(),
            "emergency_counts": zeros.copy(),
            "approaching_left_turn_counts": left,
            "queued_left_turn_counts": left.copy(),
            "approaching_right_turn_counts": right,
            "queued_right_turn_counts": right.copy(),
            "red_elapsed_s": zeros.copy(),
            "left_red_elapsed_s": zeros.copy(),
            "right_red_elapsed_s": zeros.copy(),
            "waiting_pedestrian_counts": zeros.copy(),
            "intersection_vehicle_count": 0,
            "blocked_intersection_vehicle_count": 0,
            "active_movements": ("north_through", "south_through"),
            "green_elapsed_s": 10.0,
        }

    def test_default_startup_has_no_fixed_vehicle_green(self):
        config = build_runtime_config(CONFIG)
        self.assertTrue(
            config["movement_controller"][
                "policy_selected_initial_phase"
            ]
        )
        controller = MovementTrafficLightController(config)

        self.assertEqual(controller.phase_state, "all_red")
        self.assertEqual(controller.active_movements, frozenset())
        self.assertEqual(controller.active_phase, "none")
        self.assertIsNone(controller.pending_movements)
        self.assertEqual(controller.timer, 0.0)
        self.assertTrue(
            all(state == "red" for state in controller.states.values())
        )
        self.assertTrue(
            all(
                state == "red"
                for state in controller.left_turn_states.values()
            )
        )
        self.assertEqual(
            controller.get_active_policy_movements(),
            frozenset(),
        )

    def test_startup_infers_while_all_red_and_queues_policy_choice(self):
        config = self.startup_config()
        config["traffic_lights"]["all_red_clearance_duration_s"] = 0.2
        controller = MovementTrafficLightController(config)
        observation = self.observation(vehicles={"north": 1, "east": 1})
        observation["active_movements"] = ()
        calls = []
        controller.set_phase_observation_provider(lambda: observation)

        def scores(current):
            calls.append(
                (controller.phase_state, current["active_movements"])
            )
            return {
                movement: (
                    0.9
                    if movement == "east_through"
                    else 0.1 if movement == "north_through" else 0.0
                )
                for movement in MOVEMENT_NAMES
            }

        controller.set_movement_score_provider(scores)

        controller.update(0.01)

        self.assertEqual(calls, [("all_red", ())])
        self.assertEqual(controller.phase_state, "all_red")
        self.assertEqual(controller.active_movements, frozenset())
        self.assertEqual(
            controller.pending_movements,
            frozenset(("east_through",)),
        )
        self.assertEqual(controller.timer, 0.0)
        self.assertTrue(
            all(state == "red" for state in controller.states.values())
        )

    def test_startup_waits_for_demand_then_runs_fresh_clearance(self):
        config = self.startup_config()
        config["traffic_lights"]["all_red_clearance_duration_s"] = 0.1
        controller = MovementTrafficLightController(config)
        observation = self.observation()
        observation["active_movements"] = ()
        controller.set_phase_observation_provider(lambda: observation)
        controller.set_movement_score_provider(
            lambda current: {
                movement: 0.9 if movement == "east_through" else 0.0
                for movement in MOVEMENT_NAMES
            }
        )

        controller.update(2.0)

        self.assertEqual(controller.phase_state, "all_red")
        self.assertEqual(controller.active_movements, frozenset())
        self.assertIsNone(controller.pending_movements)
        observation["vehicle_counts"]["east"] = 1
        observation["queue_lengths"]["east"] = 1

        controller.update(0.01)

        self.assertEqual(
            controller.pending_movements,
            frozenset(("east_through",)),
        )
        self.assertEqual(controller.timer, 0.0)
        controller.update(0.09)
        self.assertEqual(controller.phase_state, "all_red")
        controller.update(0.02)

        self.assertEqual(controller.phase_state, "green")
        self.assertEqual(
            controller.active_movements,
            frozenset(("east_through",)),
        )
        self.assertEqual(controller.states["east"], "green")
        self.assertEqual(controller.states["north"], "red")

    def test_startup_accepts_low_score_right_only_demand(self):
        config = self.startup_config()
        config["traffic_lights"]["all_red_clearance_duration_s"] = 0.05
        controller = MovementTrafficLightController(config)
        observation = self.observation(
            vehicles={"east": 1},
            right={"east": 1},
        )
        observation["active_movements"] = ()
        controller.set_phase_observation_provider(lambda: observation)
        controller.set_movement_score_provider(
            lambda current: {
                movement: 0.1 if movement == "east_right" else 0.0
                for movement in MOVEMENT_NAMES
            }
        )

        controller.update(0.01)
        self.assertEqual(controller.pending_movements, frozenset())
        controller.update(0.06)

        self.assertEqual(controller.phase_state, "green")
        self.assertEqual(controller.active_movements, frozenset())
        self.assertEqual(controller.get_right_turn_state("east"), "green")

    def test_minimum_green_begins_after_initial_phase_activation(self):
        config = self.startup_config()
        timing = config["traffic_lights"]
        timing["min_green_duration_s"] = 0.2
        timing["green_extension_check_interval_s"] = 0.05
        timing["all_red_clearance_duration_s"] = 0.05
        controller = MovementTrafficLightController(config)
        observation = self.observation(vehicles={"east": 1})
        observation["active_movements"] = ()
        preferred = {"movement": "east_through"}
        controller.set_phase_observation_provider(lambda: observation)
        controller.set_movement_score_provider(
            lambda current: {
                movement: (
                    0.9 if movement == preferred["movement"] else 0.0
                )
                for movement in MOVEMENT_NAMES
            }
        )

        controller.update(0.01)
        controller.update(0.06)

        self.assertEqual(controller.phase_state, "green")
        self.assertEqual(
            controller.active_movements,
            frozenset(("east_through",)),
        )
        self.assertEqual(controller.timer, 0.0)
        observation["vehicle_counts"]["east"] = 0
        observation["queue_lengths"]["east"] = 0
        observation["vehicle_counts"]["north"] = 1
        observation["queue_lengths"]["north"] = 1
        preferred["movement"] = "north_through"

        controller.update(0.19)
        self.assertEqual(controller.phase_state, "green")
        self.assertEqual(
            controller.active_movements,
            frozenset(("east_through",)),
        )
        controller.update(0.02)

        self.assertEqual(controller.phase_state, "yellow")
        self.assertEqual(
            controller.pending_movements,
            frozenset(("north_through",)),
        )

    def test_policy_selected_initial_phase_can_be_disabled(self):
        config = self.startup_config()
        config["movement_controller"][
            "policy_selected_initial_phase"
        ] = False

        controller = MovementTrafficLightController(config)

        self.assertEqual(controller.phase_state, "green")
        self.assertEqual(
            controller.active_movements,
            frozenset(("north_through", "south_through")),
        )
        self.assertEqual(controller.states["north"], "green")
        self.assertEqual(controller.states["south"], "green")
        self.assertEqual(controller.states["east"], "red")
        self.assertEqual(controller.states["west"], "red")

    def test_policy_has_vehicle_and_pedestrian_output_schema(self):
        policy = MovementPolicy(
            [0.0] * MovementPolicy.genome_size,
            duration_bounds_s=(1.0, 10.0),
        )

        scores = policy.predict_movement_scores(self.observation())

        self.assertEqual(MovementPolicy.input_size, 79)
        self.assertEqual(len(MOVEMENT_INPUT_FEATURE_NAMES), 79)
        self.assertEqual(MovementPolicy.output_size, 16)
        self.assertEqual(MovementPolicy.genome_size, 976)
        self.assertEqual(tuple(scores), POLICY_OUTPUT_NAMES)
        self.assertEqual(set(scores.values()), {0.5})
        self.assertAlmostEqual(sum(scores.values()), 8.0)

    def test_conflict_matrix_classifies_expected_concurrent_movements(self):
        controller = MovementTrafficLightController(self.config)

        self.assertTrue(
            controller.is_conflict_free(
                {"north_through", "south_through"}
            )
        )
        self.assertFalse(
            controller.is_conflict_free({"north_left", "south_left"})
        )
        self.assertFalse(
            controller.is_conflict_free({"east_left", "west_left"})
        )
        self.assertTrue(
            controller.is_conflict_free({"north_through", "north_left"})
        )
        self.assertFalse(
            controller.is_conflict_free({"north_through", "south_left"})
        )
        self.assertFalse(
            controller.is_conflict_free({"north_through", "east_through"})
        )
        self.assertTrue(
            controller.is_conflict_free(
                {
                    "north_right",
                    "south_right",
                    "east_right",
                    "west_right",
                }
            )
        )
        self.assertTrue(
            controller.is_conflict_free(
                {"north_right", "south_through"}
            )
        )
        self.assertFalse(
            controller.is_conflict_free(
                {"north_right", "east_through"}
            )
        )
        self.assertFalse(
            controller.is_conflict_free(
                {"north_right", "south_left"}
            )
        )

    def test_four_right_outputs_are_actuated_independently(self):
        controller = MovementTrafficLightController(self.config)
        observation = self.observation(
            vehicles={"north": 1, "south": 1, "east": 1},
            right={"north": 1, "south": 1, "east": 1},
        )
        controller.set_phase_observation_provider(lambda: observation)
        controller.set_movement_score_provider(
            lambda current: {
                movement: (
                    0.9
                    if movement in ("north_right", "south_right")
                    else 0.0
                )
                for movement in MOVEMENT_NAMES
            }
        )

        controller.update(0.01)

        self.assertEqual(controller.get_right_turn_state("north"), "green")
        self.assertEqual(controller.get_right_turn_state("south"), "green")
        self.assertEqual(controller.get_right_turn_state("east"), "red")
        self.assertIn(
            "north_right",
            controller.get_active_policy_movements(),
        )

    def test_low_right_output_uses_safe_permissive_main_green(self):
        controller = MovementTrafficLightController(self.config)
        observation = self.observation(
            vehicles={"north": 1},
            right={"north": 1},
        )
        controller.set_phase_observation_provider(lambda: observation)
        controller.set_movement_score_provider(
            lambda current: {movement: 0.0 for movement in MOVEMENT_NAMES}
        )

        controller.update(0.01)

        self.assertEqual(controller.get_right_turn_state("north"), "off")
        self.assertEqual(
            controller.get_right_turn_permission_state("north"),
            "green",
        )

    def test_compatible_right_only_request_keeps_the_current_main_green(self):
        timing = self.config["traffic_lights"]
        timing["min_green_duration_s"] = 0.1
        timing["green_extension_check_interval_s"] = 0.1
        controller = MovementTrafficLightController(self.config)
        observation = self.observation(
            vehicles={"north": 1},
            right={"north": 1},
        )
        controller.set_phase_observation_provider(lambda: observation)
        controller.set_movement_score_provider(
            lambda current: {
                movement: 0.9 if movement == "north_right" else 0.0
                for movement in MOVEMENT_NAMES
            }
        )

        controller.update(0.11)

        self.assertEqual(controller.phase_state, "green")
        self.assertEqual(
            controller.active_movements,
            frozenset(("north_through", "south_through")),
        )
        self.assertEqual(controller.get_right_turn_state("north"), "green")

    def test_right_output_cannot_override_pedestrian_safety_guard(self):
        controller = MovementTrafficLightController(self.config)
        observation = self.observation(
            vehicles={"north": 1},
            right={"north": 1},
        )
        controller.set_phase_observation_provider(lambda: observation)
        controller.set_movement_score_provider(
            lambda current: {
                movement: 0.9 if movement == "north_right" else 0.0
                for movement in MOVEMENT_NAMES
            }
        )
        controller.set_right_turn_activation_guard(
            lambda direction: direction != "north"
        )

        controller.update(0.01)

        self.assertEqual(controller.get_right_turn_state("north"), "red")

    def test_right_only_request_can_clear_an_incompatible_main_phase(self):
        timing = self.config["traffic_lights"]
        timing["min_green_duration_s"] = 0.1
        timing["max_green_duration_s"] = 1.0
        timing["green_extension_check_interval_s"] = 0.1
        timing["yellow_duration_s"] = 0.1
        timing["all_red_clearance_duration_s"] = 0.05
        controller = MovementTrafficLightController(self.config)
        observation = self.observation(
            vehicles={"east": 1},
            right={"east": 1},
        )
        controller.set_phase_observation_provider(lambda: observation)
        controller.set_movement_score_provider(
            lambda current: {
                movement: 0.9 if movement == "east_right" else 0.0
                for movement in MOVEMENT_NAMES
            }
        )

        controller.update(0.11)
        self.assertEqual(controller.phase_state, "yellow")
        controller.update(0.11)
        self.assertEqual(controller.phase_state, "all_red")
        controller.update(0.06)

        self.assertEqual(controller.active_movements, frozenset())
        self.assertEqual(controller.get_right_turn_state("east"), "green")

    def test_low_score_right_only_demand_clears_an_empty_main_phase(self):
        timing = self.config["traffic_lights"]
        timing["min_green_duration_s"] = 0.1
        timing["max_green_duration_s"] = 1.0
        timing["green_extension_check_interval_s"] = 0.1
        timing["yellow_duration_s"] = 0.1
        timing["all_red_clearance_duration_s"] = 0.05
        controller = MovementTrafficLightController(self.config)
        observation = self.observation(
            vehicles={"east": 1},
            right={"east": 1},
        )
        controller.set_phase_observation_provider(lambda: observation)
        controller.set_movement_score_provider(
            lambda current: {
                movement: 0.1 if movement == "east_right" else 0.0
                for movement in MOVEMENT_NAMES
            }
        )

        controller.update(0.11)
        self.assertEqual(controller.phase_state, "yellow")
        controller.update(0.11)
        self.assertEqual(controller.phase_state, "all_red")
        controller.update(0.06)

        self.assertEqual(controller.phase_state, "green")
        self.assertEqual(controller.active_movements, frozenset())
        self.assertEqual(controller.get_right_turn_state("east"), "green")

    def test_decoder_selects_multiple_compatible_through_movements(self):
        controller = MovementTrafficLightController(self.config)
        observation = self.observation(vehicles={"north": 5, "south": 4})
        scores = {movement: 0.0 for movement in MOVEMENT_NAMES}
        scores.update({"north_through": 0.9, "south_through": 0.8})

        decoded = controller.decode_scores(scores, observation)

        self.assertEqual(
            decoded,
            frozenset(("north_through", "south_through")),
        )

    def test_low_score_cannot_remove_a_compatible_demanded_movement(self):
        controller = MovementTrafficLightController(self.config)
        observation = self.observation(
            vehicles={"north": 5, "south": 4},
        )
        scores = {movement: 0.0 for movement in MOVEMENT_NAMES}
        scores.update({"north_through": 0.9, "south_through": 0.4})

        decoded = controller.decode_scores(scores, observation)

        self.assertEqual(
            decoded,
            frozenset(("north_through", "south_through")),
        )

    def test_candidate_decoder_discards_nonmaximal_safe_subsets(self):
        controller = MovementTrafficLightController(self.config)

        candidates = controller._candidate_sets(
            {"north_through", "south_through"}
        )

        self.assertEqual(
            candidates,
            [frozenset(("north_through", "south_through"))],
        )

    def test_opposing_left_candidates_remain_separate_alternatives(self):
        controller = MovementTrafficLightController(self.config)

        candidates = controller._candidate_sets(
            {"north_left", "south_left"}
        )

        self.assertEqual(
            set(candidates),
            {
                frozenset(("north_left",)),
                frozenset(("south_left",)),
            },
        )

    def test_compatible_main_addition_does_not_trigger_clearance(self):
        timing = self.config["traffic_lights"]
        timing["min_green_duration_s"] = 0.1
        timing["green_extension_check_interval_s"] = 0.1
        controller = MovementTrafficLightController(self.config)
        controller.active_movements = frozenset(("north_through",))
        controller.active_phase = controller.encode_movements(
            controller.active_movements
        )
        controller.states["south"] = "red"
        observation = self.observation(
            vehicles={"north": 5, "south": 4},
        )
        controller.set_phase_observation_provider(lambda: observation)
        controller.set_movement_score_provider(
            lambda current: {
                movement: (
                    0.9
                    if movement == "north_through"
                    else 0.4 if movement == "south_through" else 0.0
                )
                for movement in MOVEMENT_NAMES
            }
        )

        controller.update(0.11)

        self.assertEqual(controller.phase_state, "green")
        self.assertEqual(controller.pending_movements, None)
        self.assertEqual(
            controller.active_movements,
            frozenset(("north_through", "south_through")),
        )
        self.assertEqual(controller.states["south"], "green")

    def test_decoder_selects_only_one_opposing_protected_left(self):
        controller = MovementTrafficLightController(self.config)
        for first, second in (
            ("north", "south"),
            ("east", "west"),
        ):
            with self.subTest(first=first, second=second):
                observation = self.observation(
                    vehicles={first: 3, second: 3},
                    left={first: 3, second: 3},
                )
                scores = {movement: 0.0 for movement in MOVEMENT_NAMES}
                scores.update(
                    {f"{first}_left": 0.9, f"{second}_left": 0.8}
                )

                decoded = controller.decode_scores(scores, observation)

                self.assertEqual(decoded, frozenset((f"{first}_left",)))
                self.assertTrue(controller.is_conflict_free(decoded))

    def test_decoder_never_combines_conflicting_high_scores(self):
        controller = MovementTrafficLightController(self.config)
        observation = self.observation(vehicles={"north": 5, "east": 5})
        scores = {movement: 0.0 for movement in MOVEMENT_NAMES}
        scores.update({"north_through": 0.9, "east_through": 0.8})

        decoded = controller.decode_scores(scores, observation)

        self.assertIn("north_through", decoded)
        self.assertNotIn("east_through", decoded)
        self.assertTrue(controller.is_conflict_free(decoded))

    def test_queue_pressure_outweighs_one_higher_network_score(self):
        self.config["movement_controller"]["switch_hysteresis"] = 0.15
        controller = MovementTrafficLightController(self.config)
        observation = self.observation(
            vehicles={"north": 1, "east": 10},
        )
        scores = {movement: 0.0 for movement in MOVEMENT_NAMES}
        scores.update({"north_through": 0.547, "east_through": 0.353})

        decoded = controller.decode_scores(scores, observation)

        self.assertEqual(decoded, frozenset(("east_through",)))

    def test_empty_green_gaps_out_to_a_real_competing_queue(self):
        timing = self.config["traffic_lights"]
        timing["min_green_duration_s"] = 0.2
        timing["green_extension_check_interval_s"] = 0.05
        timing["yellow_duration_s"] = 0.1
        self.config["movement_controller"]["empty_green_gap_out_s"] = 0.1
        controller = MovementTrafficLightController(self.config)
        observation = self.observation(
            vehicles={"north": 1, "east": 5},
        )
        observation["near_stop_movement_counts"] = {
            movement: int(movement == "east_through") * 5
            for movement in MOVEMENT_NAMES
        }
        observation["queued_movement_counts"] = dict(
            observation["near_stop_movement_counts"]
        )
        controller.set_phase_observation_provider(lambda: observation)
        controller.set_movement_score_provider(
            lambda current: {
                movement: (
                    0.9 if movement == "north_through"
                    else 0.1 if movement == "east_through"
                    else 0.0
                )
                for movement in MOVEMENT_NAMES
            }
        )

        controller.update(0.19)
        self.assertEqual(controller.phase_state, "green")
        controller.update(0.02)

        self.assertEqual(controller.phase_state, "yellow")
        self.assertEqual(
            controller.pending_movements,
            frozenset(("east_through",)),
        )
        self.assertEqual(controller.empty_green_gap_out_count, 1)

    def test_empty_green_gap_timer_resets_for_an_approaching_vehicle(self):
        self.config["movement_controller"]["empty_green_gap_out_s"] = 0.1
        controller = MovementTrafficLightController(self.config)
        observation = self.observation(
            vehicles={"north": 1, "east": 5},
        )
        near_stop = {movement: 0 for movement in MOVEMENT_NAMES}
        queued = {movement: 0 for movement in MOVEMENT_NAMES}
        near_stop["east_through"] = 5
        queued["east_through"] = 5
        observation["near_stop_movement_counts"] = near_stop
        observation["queued_movement_counts"] = queued
        controller.set_phase_observation_provider(lambda: observation)

        controller.update(0.06)
        self.assertAlmostEqual(controller.empty_green_elapsed, 0.06)
        near_stop["north_through"] = 1
        controller.update(0.01)
        self.assertEqual(controller.empty_green_elapsed, 0.0)
        near_stop["north_through"] = 0
        controller.update(0.06)

        self.assertEqual(controller.phase_state, "green")
        self.assertAlmostEqual(controller.empty_green_elapsed, 0.06)

    def test_empty_green_without_competing_queue_does_not_gap_out(self):
        timing = self.config["traffic_lights"]
        timing["min_green_duration_s"] = 0.1
        timing["green_extension_check_interval_s"] = 0.05
        self.config["movement_controller"]["empty_green_gap_out_s"] = 0.05
        controller = MovementTrafficLightController(self.config)
        observation = self.observation()
        observation["near_stop_movement_counts"] = {
            movement: 0 for movement in MOVEMENT_NAMES
        }
        observation["queued_movement_counts"] = {
            movement: 0 for movement in MOVEMENT_NAMES
        }
        controller.set_phase_observation_provider(lambda: observation)
        controller.set_movement_score_provider(
            lambda current: {movement: 0.0 for movement in MOVEMENT_NAMES}
        )

        controller.update(0.2)

        self.assertEqual(controller.phase_state, "green")
        self.assertEqual(controller.empty_green_gap_out_count, 0)

    def test_pedestrian_guard_masks_unsafe_candidate(self):
        controller = MovementTrafficLightController(self.config)
        controller.set_movement_activation_guard(
            lambda movements: "north_through" not in movements
        )
        observation = self.observation(vehicles={"north": 5, "south": 5})
        scores = {movement: 0.0 for movement in MOVEMENT_NAMES}
        scores.update({"north_through": 0.9, "south_through": 0.8})

        decoded = controller.decode_scores(scores, observation)

        self.assertEqual(decoded, frozenset(("south_through",)))

    def test_hysteresis_keeps_a_still_useful_active_set(self):
        self.config["movement_controller"]["switch_hysteresis"] = 0.20
        controller = MovementTrafficLightController(self.config)
        observation = self.observation(vehicles={"north": 5, "east": 5})
        scores = {movement: 0.0 for movement in MOVEMENT_NAMES}
        scores.update({"north_through": 0.70, "east_through": 0.75})

        decoded = controller.decode_scores(scores, observation)

        self.assertEqual(decoded, controller.active_movements)

    def test_max_red_fairness_overrides_network_preference(self):
        controller = MovementTrafficLightController(self.config)
        observation = self.observation(vehicles={"north": 5, "east": 5})
        controller.movement_red_elapsed["east_through"] = (
            controller.max_red_duration
        )
        scores = {movement: 0.0 for movement in MOVEMENT_NAMES}
        scores.update({"north_through": 0.95, "east_through": 0.55})

        decoded = controller.decode_scores(scores, observation)

        self.assertEqual(decoded, frozenset(("east_through",)))

    def test_max_red_fairness_selects_waiting_opposing_left(self):
        controller = MovementTrafficLightController(self.config)
        observation = self.observation(
            vehicles={"north": 3, "south": 3},
            left={"north": 3, "south": 3},
        )
        controller.movement_red_elapsed["south_left"] = (
            controller.max_red_duration
        )
        scores = {movement: 0.0 for movement in MOVEMENT_NAMES}
        scores.update({"north_left": 0.95, "south_left": 0.55})

        decoded = controller.decode_scores(scores, observation)

        self.assertEqual(decoded, frozenset(("south_left",)))

    def test_controller_preserves_minimum_yellow_and_all_red_timing(self):
        timing = self.config["traffic_lights"]
        timing["min_green_duration_s"] = 0.1
        timing["max_green_duration_s"] = 1.0
        timing["green_extension_check_interval_s"] = 0.1
        timing["yellow_duration_s"] = 0.1
        timing["all_red_clearance_duration_s"] = 0.05
        controller = MovementTrafficLightController(self.config)
        observation = self.observation(vehicles={"east": 5})
        controller.set_phase_observation_provider(lambda: observation)
        controller.set_movement_score_provider(
            lambda current: {
                movement: 0.9 if movement == "east_through" else 0.0
                for movement in MOVEMENT_NAMES
            }
        )

        controller.update(0.11)
        self.assertEqual(controller.phase_state, "yellow")
        self.assertEqual(controller.states["north"], "yellow")

        controller.update(0.11)
        self.assertEqual(controller.phase_state, "all_red")
        self.assertTrue(all(state == "red" for state in controller.states.values()))

        controller.update(0.06)
        self.assertEqual(controller.phase_state, "green")
        self.assertEqual(controller.active_movements, frozenset(("east_through",)))
        self.assertEqual(controller.states["east"], "green")

    def test_opposing_left_switch_uses_yellow_and_all_red_clearance(self):
        timing = self.config["traffic_lights"]
        timing["min_green_duration_s"] = 0.1
        timing["max_green_duration_s"] = 1.0
        timing["green_extension_check_interval_s"] = 0.1
        timing["yellow_duration_s"] = 0.1
        timing["all_red_clearance_duration_s"] = 0.05
        controller = MovementTrafficLightController(self.config)
        controller.active_movements = frozenset(("north_left",))
        controller.active_phase = controller.encode_movements(
            controller.active_movements
        )
        controller.states = {
            direction: "red" for direction in controller.DIRECTIONS
        }
        controller.left_turn_states = {
            direction: "red" for direction in controller.DIRECTIONS
        }
        controller.left_turn_states["north"] = "green"
        observation = self.observation(
            vehicles={"north": 3, "south": 3},
            left={"north": 3, "south": 3},
        )
        observation["active_movements"] = ("north_left",)
        controller.set_phase_observation_provider(lambda: observation)
        controller.set_movement_score_provider(
            lambda current: {
                movement: (
                    0.9
                    if movement == "south_left"
                    else 0.8 if movement == "north_left" else 0.0
                )
                for movement in MOVEMENT_NAMES
            }
        )

        controller.update(0.11)
        self.assertEqual(controller.phase_state, "yellow")
        self.assertEqual(
            controller.pending_movements,
            frozenset(("south_left",)),
        )
        self.assertEqual(controller.left_turn_states["north"], "yellow")
        self.assertEqual(controller.left_turn_states["south"], "red")

        controller.update(0.11)
        self.assertEqual(controller.phase_state, "all_red")
        self.assertTrue(
            all(
                state == "red"
                for state in controller.left_turn_states.values()
            )
        )

        controller.update(0.06)
        self.assertEqual(controller.phase_state, "green")
        self.assertEqual(
            controller.active_movements,
            frozenset(("south_left",)),
        )
        self.assertEqual(controller.left_turn_states["north"], "red")
        self.assertEqual(controller.left_turn_states["south"], "green")

    def test_parallel_training_matches_sequential_training(self):
        options = {
            "duration_bounds_s": (0.1, 0.3),
            "population_size": 2,
            "generations": 1,
            "seeds": (1,),
            "evaluation_duration_s": 0.1,
            "speed_factor": 2.0,
            "traffic_profiles": (),
            "random_seed": 17,
        }

        sequential = MovementPolicyEvolution(
            self.config,
            workers=1,
            **options,
        ).run()
        parallel = MovementPolicyEvolution(
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


if __name__ == "__main__":
    unittest.main()
