import unittest

from config import CONFIG, build_runtime_config
from main_movement_policy import load_movement_policy
from simulation.movement_neuroevolution import (
    MOVEMENT_INPUT_FEATURE_NAMES,
    MOVEMENT_NAMES,
    VehicleMovementPolicy,
)
from simulation.traffic_light import MovementTrafficLightController


class MovementPolicyTests(unittest.TestCase):
    def setUp(self):
        self.config = build_runtime_config(CONFIG)
        self.config["traffic_lights"]["min_green_duration_s"] = 1.0
        self.config["traffic_lights"]["yellow_duration_s"] = 0.2
        self.config["traffic_lights"]["all_red_clearance_duration_s"] = 0.1

    def observation(self, **overrides):
        zeros = {direction: 0 for direction in ("north", "south", "east", "west")}
        observation = {
            "vehicle_counts": zeros.copy(),
            "queue_lengths": zeros.copy(),
            "average_speed_ratios": zeros.copy(),
            "emergency_counts": zeros.copy(),
            "approaching_left_turn_counts": zeros.copy(),
            "queued_left_turn_counts": zeros.copy(),
            "approaching_right_turn_counts": zeros.copy(),
            "queued_right_turn_counts": zeros.copy(),
            "red_elapsed_s": zeros.copy(),
            "left_red_elapsed_s": zeros.copy(),
            "right_red_elapsed_s": zeros.copy(),
            "intersection_vehicle_count": 0,
            "blocked_intersection_vehicle_count": 0,
            "active_movements": (),
            "green_elapsed_s": 0.0,
        }
        observation.update(overrides)
        return observation

    def test_published_network_schema(self):
        self.assertEqual(len(MOVEMENT_INPUT_FEATURE_NAMES), 59)
        self.assertEqual(VehicleMovementPolicy.input_size, 59)
        self.assertEqual(VehicleMovementPolicy.hidden_size, 10)
        self.assertEqual(VehicleMovementPolicy.output_size, 12)
        self.assertEqual(VehicleMovementPolicy.genome_size, 732)

    def test_public_model_matches_schema(self):
        policy = load_movement_policy()
        scores = policy.predict_movement_scores(self.observation())
        self.assertEqual(tuple(scores), MOVEMENT_NAMES)
        self.assertTrue(all(0.0 <= score <= 1.0 for score in scores.values()))

    def test_controller_starts_all_red_until_policy_selects_demand(self):
        controller = MovementTrafficLightController(self.config)
        self.assertEqual(controller.phase_state, "all_red")
        self.assertFalse(controller.get_active_policy_movements())

    def test_expected_movement_conflicts(self):
        controller = MovementTrafficLightController
        self.assertFalse(
            controller.movements_conflict("north_through", "south_through")
        )
        self.assertTrue(
            controller.movements_conflict("north_through", "east_through")
        )
        self.assertTrue(
            controller.movements_conflict("north_left", "south_left")
        )

    def test_decoder_combines_compatible_opposing_through_movements(self):
        controller = MovementTrafficLightController(self.config)
        observation = self.observation(
            vehicle_counts={"north": 2, "south": 2, "east": 0, "west": 0},
            queue_lengths={"north": 2, "south": 2, "east": 0, "west": 0},
        )
        scores = {movement: 0.0 for movement in MOVEMENT_NAMES}
        scores["north_through"] = scores["south_through"] = 1.0
        decoded = controller.decode_scores(scores, observation)
        self.assertEqual(
            decoded,
            frozenset(("north_through", "south_through")),
        )

    def test_decoder_never_combines_opposing_protected_lefts(self):
        controller = MovementTrafficLightController(self.config)
        observation = self.observation(
            vehicle_counts={"north": 1, "south": 1, "east": 0, "west": 0},
            approaching_left_turn_counts={
                "north": 1,
                "south": 1,
                "east": 0,
                "west": 0,
            },
            queued_left_turn_counts={
                "north": 1,
                "south": 1,
                "east": 0,
                "west": 0,
            },
        )
        scores = {movement: 0.0 for movement in MOVEMENT_NAMES}
        scores["north_left"] = scores["south_left"] = 1.0
        decoded = controller.decode_scores(scores, observation)
        self.assertEqual(len(decoded.intersection(("north_left", "south_left"))), 1)


if __name__ == "__main__":
    unittest.main()
