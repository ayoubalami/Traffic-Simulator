import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from simulation.movement_neuroevolution import (
    MOVEMENT_INPUT_FEATURE_NAMES,
    MOVEMENT_NAMES,
    MOVEMENT_POLICY_FORMAT_VERSION,
    PEDESTRIAN_OUTPUT_NAMES,
    POLICY_OUTPUT_NAMES,
    MovementPolicy,
)
from train_movement_policy import load_warm_start_policy, parse_arguments


class MovementTrainingCliTests(unittest.TestCase):
    def parse(self, *arguments):
        with patch.object(
            sys,
            "argv",
            ["train_movement_policy.py", *arguments],
        ):
            return parse_arguments()

    def test_defaults_use_fixed_physics_timestep_and_diagonal_es(self):
        arguments = self.parse()

        self.assertEqual(arguments.optimizer, "diagonal-es")
        self.assertAlmostEqual(arguments.speed_factor, 1.0)
        self.assertAlmostEqual(arguments.timestep, 1 / 30)
        self.assertIsNone(arguments.warm_start)
        self.assertIsNone(arguments.checkpoint)
        self.assertIsNone(arguments.resume)
        self.assertIsNone(arguments.anchor_scenarios_count)
        self.assertEqual(arguments.observation_mode, "configured")

    def test_explicit_observation_mode_parses(self):
        arguments = self.parse("--observation-mode", "full-state")

        self.assertEqual(arguments.observation_mode, "full-state")

    def test_es_and_staged_evaluation_options_parse(self):
        arguments = self.parse(
            "--optimizer",
            "diagonal-es",
            "--warm-start",
            "models/warm.json",
            "--checkpoint",
            "models/training.checkpoint.json",
            "--initial-sigma",
            "0.12",
            "--sigma-min",
            "0.015",
            "--sigma-max",
            "0.8",
            "--elite-fraction",
            "0.2",
            "--distribution-learning-rate",
            "0.35",
            "--stagnation-patience",
            "9",
            "--reheat-factor",
            "1.7",
            "--screen-duration",
            "30",
            "--screen-scenarios",
            "2",
            "--promotion-fraction",
            "0.25",
            "--promotion-duration",
            "90",
            "--promotion-scenarios",
            "4",
            "--anchor-interval",
            "5",
            "--anchor-candidates",
            "3",
            "--anchor-scenarios-count",
            "7",
            "--robustness-penalty",
            "0.4",
            "--timestep",
            "0.02",
            "--speed-factor",
            "1",
        )

        self.assertEqual(arguments.optimizer, "diagonal-es")
        self.assertEqual(arguments.warm_start, Path("models/warm.json"))
        self.assertEqual(
            arguments.checkpoint,
            Path("models/training.checkpoint.json"),
        )
        self.assertAlmostEqual(arguments.initial_sigma, 0.12)
        self.assertAlmostEqual(arguments.sigma_min, 0.015)
        self.assertAlmostEqual(arguments.sigma_max, 0.8)
        self.assertAlmostEqual(arguments.elite_fraction, 0.2)
        self.assertAlmostEqual(arguments.distribution_learning_rate, 0.35)
        self.assertEqual(arguments.stagnation_patience, 9)
        self.assertAlmostEqual(arguments.reheat_factor, 1.7)
        self.assertAlmostEqual(arguments.screen_duration, 30.0)
        self.assertEqual(arguments.screen_scenarios, 2)
        self.assertAlmostEqual(arguments.promotion_fraction, 0.25)
        self.assertAlmostEqual(arguments.promotion_duration, 90.0)
        self.assertEqual(arguments.promotion_scenarios, 4)
        self.assertEqual(arguments.anchor_interval, 5)
        self.assertEqual(arguments.anchor_candidates, 3)
        self.assertEqual(arguments.anchor_scenarios_count, 7)
        self.assertAlmostEqual(arguments.robustness_penalty, 0.4)
        self.assertAlmostEqual(arguments.timestep, 0.02)
        self.assertAlmostEqual(arguments.speed_factor, 1.0)

    def test_resume_can_write_to_a_checkpoint(self):
        arguments = self.parse(
            "--resume",
            "models/previous.checkpoint.json",
            "--checkpoint",
            "models/next.checkpoint.json",
        )

        self.assertEqual(
            arguments.resume,
            Path("models/previous.checkpoint.json"),
        )
        self.assertEqual(
            arguments.checkpoint,
            Path("models/next.checkpoint.json"),
        )
        self.assertIsNone(arguments.warm_start)

    def test_warm_start_and_resume_are_mutually_exclusive(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                self.parse(
                    "--warm-start",
                    "models/warm.json",
                    "--resume",
                    "models/checkpoint.json",
                )

        self.assertEqual(raised.exception.code, 2)

    def test_warm_start_loader_validates_and_reuses_current_weights(self):
        data = {
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
            "weights": [0.125] * MovementPolicy.genome_size,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            path.write_text(json.dumps(data), encoding="utf-8")

            policy = load_warm_start_policy(path, (5.0, 30.0), 60.0)

            self.assertEqual(policy.weights, data["weights"])
            self.assertEqual(policy.minimum_duration_s, 5.0)
            self.assertEqual(policy.maximum_duration_s, 30.0)

            data["policy_type"] = "six_phase"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "compatible movement policy"):
                load_warm_start_policy(path, (5.0, 30.0), 60.0)


if __name__ == "__main__":
    unittest.main()
