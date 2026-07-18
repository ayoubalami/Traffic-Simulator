import json
import tempfile
import unittest
from pathlib import Path

from main_movement_policy import load_movement_policy
from simulation.movement_neuroevolution import (
    LEGACY_MOVEMENT_INPUT_FEATURE_NAMES,
    LEGACY_MOVEMENT_POLICY_FORMAT_VERSION,
    MOVEMENT_INPUT_FEATURE_NAMES,
    MOVEMENT_NAMES,
    MOVEMENT_POLICY_FORMAT_VERSION,
    PEDESTRIAN_OUTPUT_NAMES,
    POLICY_OUTPUT_NAMES,
    MovementPolicy,
    migrate_legacy_movement_policy_weights,
)
from train_movement_policy import load_warm_start_policy


class PedestrianPolicySchemaTests(unittest.TestCase):
    def make_policy(self):
        return MovementPolicy(
            [0.0] * MovementPolicy.genome_size,
            duration_bounds_s=(5.0, 30.0),
            max_red_duration_s=60.0,
        )

    def model_data(self):
        return {
            "format_version": MOVEMENT_POLICY_FORMAT_VERSION,
            "policy_type": "movement_multi_hot",
            "movements": list(MOVEMENT_NAMES),
            "pedestrian_outputs": list(PEDESTRIAN_OUTPUT_NAMES),
            "outputs": list(POLICY_OUTPUT_NAMES),
            "network": {
                "input_size": MovementPolicy.input_size,
                "input_features": list(MOVEMENT_INPUT_FEATURE_NAMES),
                "hidden_size": MovementPolicy.hidden_size,
                "output_size": MovementPolicy.output_size,
                "output_names": list(POLICY_OUTPUT_NAMES),
            },
            "duration_bounds_s": [5.0, 30.0],
            "max_red_duration_s": 60.0,
            "pedestrian_decoder": {"max_walk_duration_s": 12.0},
            "weights": [0.0] * MovementPolicy.genome_size,
        }

    def legacy_model_data(self):
        legacy_genome_size = (
            (len(LEGACY_MOVEMENT_INPUT_FEATURE_NAMES) + 1)
            * MovementPolicy.hidden_size
            + (MovementPolicy.hidden_size + 1) * len(MOVEMENT_NAMES)
        )
        return {
            "format_version": LEGACY_MOVEMENT_POLICY_FORMAT_VERSION,
            "policy_type": "movement_multi_hot",
            "movements": list(MOVEMENT_NAMES),
            "network": {
                "input_size": len(LEGACY_MOVEMENT_INPUT_FEATURE_NAMES),
                "input_features": list(LEGACY_MOVEMENT_INPUT_FEATURE_NAMES),
                "hidden_size": MovementPolicy.hidden_size,
                "output_size": len(MOVEMENT_NAMES),
            },
            "duration_bounds_s": [5.0, 30.0],
            "max_red_duration_s": 60.0,
            "weights": [float(index) for index in range(legacy_genome_size)],
        }

    def test_network_is_79_by_10_by_16_with_explicit_output_order(self):
        policy = self.make_policy()

        scores = policy.predict_movement_scores({})

        self.assertEqual(MovementPolicy.input_size, 79)
        self.assertEqual(MovementPolicy.hidden_size, 10)
        self.assertEqual(MovementPolicy.output_size, 16)
        self.assertEqual(MovementPolicy.genome_size, 976)
        self.assertEqual(
            PEDESTRIAN_OUTPUT_NAMES,
            ("north_walk", "south_walk", "east_walk", "west_walk"),
        )
        self.assertEqual(POLICY_OUTPUT_NAMES[:12], MOVEMENT_NAMES)
        self.assertEqual(tuple(scores), POLICY_OUTPUT_NAMES)
        self.assertEqual(policy.last_output_scores, scores)
        self.assertEqual(policy.last_movement_scores, scores)

    def test_crosswalk_features_are_normalized_independently(self):
        policy = self.make_policy()
        observation = {
            "active_crossing_pedestrian_counts": {
                "north": 5,
                "south": 10,
                "east": 20,
                "west": 0,
            },
            "crosswalk_vehicle_occupancy_counts": {
                "north": 1,
                "south": 5,
                "east": 8,
                "west": 0,
            },
            "pedestrian_red_elapsed_s": {
                "north": 30.0,
                "south": 60.0,
                "east": 90.0,
                "west": 0.0,
            },
            "active_pedestrian_walks": {
                "north": "green",
                "south": "red",
                "east": True,
                "west": False,
            },
        }

        inputs = policy._build_inputs(observation)
        by_name = dict(zip(MOVEMENT_INPUT_FEATURE_NAMES, inputs))

        self.assertEqual(
            [
                by_name[f"active_crossing_pedestrian_count_{direction}"]
                for direction in ("north", "south", "east", "west")
            ],
            [0.5, 1.0, 1.0, 0.0],
        )
        self.assertEqual(
            [
                by_name[f"crosswalk_vehicle_occupancy_count_{direction}"]
                for direction in ("north", "south", "east", "west")
            ],
            [0.2, 1.0, 1.0, 0.0],
        )
        self.assertEqual(
            [
                by_name[f"pedestrian_red_elapsed_{direction}"]
                for direction in ("north", "south", "east", "west")
            ],
            [0.5, 1.0, 1.0, 0.0],
        )
        self.assertEqual(
            [
                by_name[f"active_pedestrian_walk_{direction}"]
                for direction in ("north", "south", "east", "west")
            ],
            [1.0, 0.0, 1.0, 0.0],
        )

    def test_deployment_and_warm_start_loaders_require_output_order(self):
        data = self.model_data()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            path.write_text(json.dumps(data), encoding="utf-8")

            deployed = load_movement_policy(path)
            warm_start = load_warm_start_policy(path, (5.0, 30.0), 60.0)

            self.assertEqual(len(deployed.weights), 976)
            self.assertEqual(len(warm_start.weights), 976)
            self.assertEqual(
                deployed.pedestrian_decoder_config[
                    "max_walk_duration_s"
                ],
                12.0,
            )

            data["outputs"][-2:] = reversed(data["outputs"][-2:])
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "output order"):
                load_movement_policy(path)
            with self.assertRaisesRegex(ValueError, "incompatible"):
                load_warm_start_policy(path, (5.0, 30.0), 60.0)

    def test_format_two_baseline_is_zero_expanded_but_stays_vehicle_only(self):
        data = self.legacy_model_data()
        migrated = migrate_legacy_movement_policy_weights(data["weights"])
        old_input_size = len(LEGACY_MOVEMENT_INPUT_FEATURE_NAMES)
        insertion_index = LEGACY_MOVEMENT_INPUT_FEATURE_NAMES.index(
            "intersection_vehicle_count"
        )

        old_first_hidden = data["weights"][: old_input_size + 1]
        new_first_hidden = migrated[: MovementPolicy.input_size + 1]
        self.assertEqual(
            new_first_hidden[:insertion_index],
            old_first_hidden[:insertion_index],
        )
        self.assertEqual(
            new_first_hidden[insertion_index : insertion_index + 16],
            [0.0] * 16,
        )
        self.assertEqual(
            new_first_hidden[insertion_index + 16 : -1],
            old_first_hidden[insertion_index:-1],
        )
        self.assertEqual(new_first_hidden[-1], old_first_hidden[-1])
        self.assertEqual(migrated[-44:], [0.0] * 44)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-policy.json"
            path.write_text(json.dumps(data), encoding="utf-8")

            deployed = load_movement_policy(path)
            warm_start = load_warm_start_policy(path, (5.0, 30.0), 60.0)

        self.assertTrue(deployed.legacy_vehicle_only)
        self.assertFalse(warm_start.legacy_vehicle_only)
        self.assertEqual(tuple(deployed.predict_movement_scores({})), MOVEMENT_NAMES)
        self.assertEqual(tuple(deployed.last_output_scores), MOVEMENT_NAMES)
        self.assertEqual(
            tuple(warm_start.predict_movement_scores({})),
            POLICY_OUTPUT_NAMES,
        )


if __name__ == "__main__":
    unittest.main()
