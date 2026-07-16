import unittest

from config import CONFIG, build_runtime_config
from simulation import MovementPolicy, MovementPolicyEvolution
from simulation.movement_neuroevolution import (
    MOVEMENT_INPUT_FEATURE_NAMES,
    MOVEMENT_NAMES,
)
from simulation.traffic_light import MovementTrafficLightController


class MovementPolicyTests(unittest.TestCase):
    def setUp(self):
        self.config = build_runtime_config(CONFIG)
        self.config["movement_controller"]["switch_hysteresis"] = 0.0

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

    def test_policy_has_independent_eight_output_schema(self):
        policy = MovementPolicy(
            [0.0] * MovementPolicy.genome_size,
            duration_bounds_s=(1.0, 10.0),
        )

        scores = policy.predict_movement_scores(self.observation())

        self.assertEqual(MovementPolicy.input_size, 59)
        self.assertEqual(len(MOVEMENT_INPUT_FEATURE_NAMES), 59)
        self.assertEqual(MovementPolicy.output_size, 8)
        self.assertEqual(MovementPolicy.genome_size, 688)
        self.assertEqual(tuple(scores), MOVEMENT_NAMES)
        self.assertEqual(set(scores.values()), {0.5})
        self.assertAlmostEqual(sum(scores.values()), 4.0)

    def test_conflict_matrix_allows_expected_concurrent_movements(self):
        controller = MovementTrafficLightController(self.config)

        self.assertTrue(
            controller.is_conflict_free(
                {"north_through", "south_through"}
            )
        )
        self.assertTrue(
            controller.is_conflict_free({"north_left", "south_left"})
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

    def test_decoder_can_pair_opposing_protected_lefts(self):
        controller = MovementTrafficLightController(self.config)
        observation = self.observation(
            vehicles={"north": 3, "south": 3},
            left={"north": 3, "south": 3},
        )
        scores = {movement: 0.0 for movement in MOVEMENT_NAMES}
        scores.update({"north_left": 0.9, "south_left": 0.9})

        decoded = controller.decode_scores(scores, observation)

        self.assertEqual(decoded, frozenset(("north_left", "south_left")))

    def test_decoder_never_combines_conflicting_high_scores(self):
        controller = MovementTrafficLightController(self.config)
        observation = self.observation(vehicles={"north": 5, "east": 5})
        scores = {movement: 0.0 for movement in MOVEMENT_NAMES}
        scores.update({"north_through": 0.9, "east_through": 0.8})

        decoded = controller.decode_scores(scores, observation)

        self.assertIn("north_through", decoded)
        self.assertNotIn("east_through", decoded)
        self.assertTrue(controller.is_conflict_free(decoded))

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
