import json
import tempfile
import unittest
from pathlib import Path

from config import (
    CONFIG,
    VEHICLES_ONLY_SCOPE,
    apply_movement_control_scope,
    build_runtime_config,
    movement_fitness_weights_for_scope,
)
from main_movement_policy import load_movement_policy
from simulation import Simulation
from simulation.evaluation import _apply_policy_control_scope
from simulation.movement_neuroevolution import (
    LEGACY_MOVEMENT_INPUT_FEATURE_NAMES,
    LEGACY_MOVEMENT_POLICY_FORMAT_VERSION,
    MOVEMENT_INPUT_FEATURE_NAMES,
    MOVEMENT_NAMES,
    MOVEMENT_POLICY_FORMAT_VERSION,
    PEDESTRIAN_OUTPUT_NAMES,
    POLICY_OUTPUT_NAMES,
    VEHICLE_ONLY_INPUT_FEATURE_NAMES,
    VEHICLE_ONLY_POLICY_FORMAT_VERSION,
    MovementPolicy,
    VehicleMovementPolicy,
    VehicleMovementPolicyEvolution,
)
from train_movement_policy import load_warm_start_policy


class VehicleOnlyPolicyTests(unittest.TestCase):
    @staticmethod
    def make_policy():
        return VehicleMovementPolicy(
            [0.0] * VehicleMovementPolicy.genome_size,
            duration_bounds_s=(1.0, 10.0),
            max_red_duration_s=60.0,
        )

    def test_network_schema_contains_only_vehicle_control_signals(self):
        policy = self.make_policy()

        scores = policy.predict_movement_scores({})

        self.assertIsInstance(VEHICLE_ONLY_POLICY_FORMAT_VERSION, int)
        self.assertGreater(VEHICLE_ONLY_POLICY_FORMAT_VERSION, 0)
        self.assertEqual(len(VEHICLE_ONLY_INPUT_FEATURE_NAMES), 59)
        self.assertEqual(VehicleMovementPolicy.input_size, 59)
        self.assertEqual(VehicleMovementPolicy.hidden_size, 10)
        self.assertEqual(VehicleMovementPolicy.output_size, len(MOVEMENT_NAMES))
        self.assertEqual(VehicleMovementPolicy.output_size, 12)
        self.assertEqual(VehicleMovementPolicy.genome_size, 732)
        self.assertEqual(tuple(scores), MOVEMENT_NAMES)
        self.assertEqual(set(scores.values()), {0.5})
        self.assertEqual(policy.last_output_scores, scores)
        self.assertEqual(policy.last_movement_scores, scores)
        self.assertIs(
            VehicleMovementPolicyEvolution.policy_class,
            VehicleMovementPolicy,
        )

        for feature in VEHICLE_ONLY_INPUT_FEATURE_NAMES:
            self.assertNotIn("pedestrian", feature)
            self.assertNotIn("crosswalk", feature)

    def test_disabled_pedestrians_never_spawn_or_receive_walk(self):
        config = build_runtime_config(CONFIG)
        config.setdefault("road_users", {})["pedestrians_enabled"] = False
        # A non-zero population cap and immediate spawn interval prove the
        # road-user switch is authoritative rather than relying on max_active.
        config["pedestrian_defaults"]["max_active"] = 8
        config["pedestrian_defaults"]["spawn_interval_min"] = 0.0
        config["pedestrian_defaults"]["spawn_interval_max"] = 0.0
        config["movement_controller"]["policy_selected_initial_phase"] = False
        config["simulation"]["arrival_rates_per_s"] = {
            direction: 0.0
            for direction in ("north", "south", "east", "west")
        }
        policy = self.make_policy()
        simulation = Simulation(
            config,
            random_seed=7,
            movement_score_provider=policy.predict_movement_scores,
        )

        for _ in range(5):
            simulation.update(0.1)

        self.assertEqual(simulation.pedestrians, [])
        self.assertTrue(
            all(
                simulation.light_controller.get_pedestrian_state(direction)
                == "red"
                for direction in ("north", "south", "east", "west")
            )
        )

    def test_vehicle_only_scope_excludes_pedestrian_fitness_objectives(self):
        config = build_runtime_config(CONFIG)

        apply_movement_control_scope(config, VEHICLES_ONLY_SCOPE)
        fitness, six_phase_fitness = movement_fitness_weights_for_scope(
            config,
            VEHICLES_ONLY_SCOPE,
        )

        self.assertFalse(config["road_users"]["pedestrians_enabled"])
        self.assertEqual(config["pedestrian_defaults"]["max_active"], 0)
        self.assertEqual(
            fitness["avg_pedestrian_wait_time_penalty"],
            0.0,
        )
        self.assertEqual(
            six_phase_fitness[
                "wasted_pedestrian_walk_fraction_penalty"
            ],
            0.0,
        )
        self.assertGreater(
            config["fitness"]["avg_pedestrian_wait_time_penalty"],
            0.0,
        )

    def test_evaluation_applies_the_policy_scope_without_mutating_config(self):
        config = build_runtime_config(CONFIG)
        config["road_users"]["pedestrians_enabled"] = True
        config["pedestrian_defaults"]["max_active"] = 5
        policy = self.make_policy()

        args, kwargs = _apply_policy_control_scope(
            (config, policy),
            {},
            policy,
        )
        scoped = args[0]

        self.assertEqual(kwargs, {})
        self.assertTrue(config["road_users"]["pedestrians_enabled"])
        self.assertEqual(config["pedestrian_defaults"]["max_active"], 5)
        self.assertFalse(scoped["road_users"]["pedestrians_enabled"])
        self.assertEqual(scoped["pedestrian_defaults"]["max_active"], 5)

    def test_format_four_loader_preserves_vehicle_only_scope(self):
        data = {
            "format_version": VEHICLE_ONLY_POLICY_FORMAT_VERSION,
            "policy_type": "movement_multi_hot",
            "control_scope": VEHICLES_ONLY_SCOPE,
            "movements": list(MOVEMENT_NAMES),
            "pedestrian_outputs": [],
            "outputs": list(MOVEMENT_NAMES),
            "network": {
                "input_size": VehicleMovementPolicy.input_size,
                "input_features": list(VEHICLE_ONLY_INPUT_FEATURE_NAMES),
                "hidden_size": VehicleMovementPolicy.hidden_size,
                "output_size": VehicleMovementPolicy.output_size,
                "output_names": list(MOVEMENT_NAMES),
            },
            "duration_bounds_s": [5.0, 30.0],
            "max_red_duration_s": 60.0,
            "pedestrian_decoder": {"enabled": True},
            "weights": [0.0] * VehicleMovementPolicy.genome_size,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vehicle-policy.json"
            path.write_text(json.dumps(data), encoding="utf-8")

            policy = load_movement_policy(path)
            warm_start = load_warm_start_policy(
                path,
                (5.0, 30.0),
                60.0,
                VEHICLES_ONLY_SCOPE,
            )

        self.assertIsInstance(policy, VehicleMovementPolicy)
        self.assertIsInstance(warm_start, VehicleMovementPolicy)
        self.assertEqual(policy.control_scope, VEHICLES_ONLY_SCOPE)
        self.assertEqual(policy.pedestrian_decoder_config, {})

    def test_combined_warm_start_can_be_projected_to_vehicle_only(self):
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
            "weights": [0.25] * MovementPolicy.genome_size,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "combined-policy.json"
            path.write_text(json.dumps(data), encoding="utf-8")

            policy = load_warm_start_policy(
                path,
                (5.0, 30.0),
                60.0,
                VEHICLES_ONLY_SCOPE,
            )

        self.assertIsInstance(policy, VehicleMovementPolicy)
        self.assertEqual(len(policy.weights), 732)
        self.assertEqual(set(policy.weights), {0.25})

    def test_legacy_warm_start_can_be_projected_to_vehicle_only(self):
        legacy_genome_size = (
            (len(LEGACY_MOVEMENT_INPUT_FEATURE_NAMES) + 1)
            * VehicleMovementPolicy.hidden_size
            + (VehicleMovementPolicy.hidden_size + 1) * len(MOVEMENT_NAMES)
        )
        data = {
            "format_version": LEGACY_MOVEMENT_POLICY_FORMAT_VERSION,
            "policy_type": "movement_multi_hot",
            "movements": list(MOVEMENT_NAMES),
            "network": {
                "input_size": len(LEGACY_MOVEMENT_INPUT_FEATURE_NAMES),
                "input_features": list(LEGACY_MOVEMENT_INPUT_FEATURE_NAMES),
                "hidden_size": VehicleMovementPolicy.hidden_size,
                "output_size": len(MOVEMENT_NAMES),
            },
            "weights": [0.75] * legacy_genome_size,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-policy.json"
            path.write_text(json.dumps(data), encoding="utf-8")

            policy = load_warm_start_policy(
                path,
                (5.0, 30.0),
                60.0,
                VEHICLES_ONLY_SCOPE,
            )

        self.assertIsInstance(policy, VehicleMovementPolicy)
        self.assertEqual(len(policy.weights), 732)
        self.assertEqual(set(policy.weights), {0.75})


if __name__ == "__main__":
    unittest.main()
