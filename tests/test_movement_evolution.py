import math
import random
import tempfile
from pathlib import Path
import unittest

from config import CONFIG, build_runtime_config
from simulation.movement_neuroevolution import (
    MovementPolicy,
    MovementPolicyEvolution,
    _candidate_from_evaluation,
)


class DeterministicMovementEvolution(MovementPolicyEvolution):
    """Cheap deterministic evaluator used to exercise optimizer behavior."""

    def __init__(self, *args, **kwargs):
        self.stage_calls = []
        super().__init__(*args, **kwargs)

    def _evaluate_policy(self, policy, duration_s, scenario_pairs, stage):
        scenario_names = tuple(
            (profile.get("name", "unnamed"), seed)
            for profile, seed in scenario_pairs
        )
        self.stage_calls.append((stage, duration_s, scenario_names))
        distance = sum((weight - 0.15) ** 2 for weight in policy.weights[:8])
        evaluations = []
        for profile, seed in scenario_pairs:
            evaluations.append(
                {
                    "fitness": (
                        10.0
                        - distance
                        + float(profile.get("fitness_offset", 0.0))
                        + int(seed) * 0.001
                    )
                }
            )
        evaluation = {
            "mean_fitness": sum(item["fitness"] for item in evaluations)
            / len(evaluations),
            "mean_metrics": {},
            "evaluations": evaluations,
            "requested_scenarios": scenario_names,
            "evaluated_scenarios": scenario_names,
            "skipped_scenarios": (),
            "skipped_scenario_count": 0,
        }
        return _candidate_from_evaluation(
            policy,
            evaluation,
            self.robustness_penalty,
            stage,
        )


class MovementEvolutionTests(unittest.TestCase):
    def setUp(self):
        self.profiles = (
            {"name": "balanced", "fitness_offset": 0.0},
            {"name": "peak", "fitness_offset": -0.5},
        )
        self.config = {
            "traffic_lights": {"max_red_duration_s": 60.0},
            "six_phase_training": {"traffic_profiles": list(self.profiles)},
        }

    def trainer(self, trainer_class=DeterministicMovementEvolution, **overrides):
        options = {
            "optimizer": "diagonal-es",
            "population_size": 6,
            "generations": 2,
            "seeds": (1, 2),
            "evaluation_duration_s": 2.0,
            "timestep_s": 0.1,
            "speed_factor": 1.0,
            "traffic_profiles": self.profiles,
            "workers": 1,
            "random_seed": 17,
            "initial_sigma": 0.12,
            "screening_duration_s": 0.5,
            "screening_scenarios": 1,
            "promotion_duration_s": 1.0,
            "promotion_scenarios": 2,
            "anchor_interval": 1,
            "anchor_candidates": 2,
        }
        options.update(overrides)
        return trainer_class(self.config, **options)

    def test_xavier_initialization_uses_zero_biases(self):
        policy = MovementPolicy.random(random.Random(3), (1.0, 10.0), 60.0)
        hidden_limit = math.sqrt(
            6.0 / (MovementPolicy.input_size + MovementPolicy.hidden_size)
        )
        output_limit = math.sqrt(
            6.0 / (MovementPolicy.hidden_size + MovementPolicy.output_size)
        )

        cursor = 0
        for _ in range(MovementPolicy.hidden_size):
            layer_weights = policy.weights[cursor : cursor + MovementPolicy.input_size]
            self.assertTrue(all(abs(value) <= hidden_limit for value in layer_weights))
            cursor += MovementPolicy.input_size
            self.assertEqual(policy.weights[cursor], 0.0)
            cursor += 1
        for _ in range(MovementPolicy.output_size):
            layer_weights = policy.weights[cursor : cursor + MovementPolicy.hidden_size]
            self.assertTrue(all(abs(value) <= output_limit for value in layer_weights))
            cursor += MovementPolicy.hidden_size
            self.assertEqual(policy.weights[cursor], 0.0)
            cursor += 1
        self.assertEqual(cursor, MovementPolicy.genome_size)

    def test_population_contains_mirrored_pairs_around_the_mean(self):
        trainer = self.trainer()
        trainer._initialize_distribution()

        population = trainer._sample_population()

        self.assertEqual(population[0].weights, trainer._distribution_mean)
        for positive_index, negative_index in ((1, 2), (3, 4)):
            for positive, negative, mean in zip(
                population[positive_index].weights,
                population[negative_index].weights,
                trainer._distribution_mean,
            ):
                self.assertAlmostEqual((positive + negative) / 2.0, mean)

    def test_distribution_update_is_rank_based_and_bounded(self):
        trainer = self.trainer()
        trainer._initialize_distribution()
        policies = [
            MovementPolicy(
                [value] * MovementPolicy.genome_size,
                trainer.duration_bounds_s,
                trainer.max_red_duration_s,
            )
            for value in (0.3, 0.2, -0.2, -0.3)
        ]
        promoted = [
            {"policy": policy, "fitness": fitness}
            for policy, fitness in zip(policies, (4.0, 3.0, 2.0, 1.0))
        ]

        updated = trainer._update_distribution(promoted)

        self.assertTrue(updated)
        self.assertGreater(trainer._distribution_mean[0], 0.0)
        self.assertTrue(
            all(
                trainer.sigma_min <= sigma <= trainer.sigma_max
                for sigma in trainer._distribution_sigma
            )
        )
        before = list(trainer._distribution_mean)
        self.assertFalse(
            trainer._update_distribution(
                [{"policy": policy, "fitness": 1.0} for policy in policies]
            )
        )
        self.assertEqual(trainer._distribution_mean, before)

    def test_stages_rotate_common_batches_and_keep_anchor_fixed(self):
        trainer = self.trainer(generations=2)

        result = trainer.run()

        first_screen = trainer.stage_calls[:6]
        first_promotion = trainer.stage_calls[6:8]
        first_anchor = trainer.stage_calls[8:10]
        second_screen = trainer.stage_calls[10:16]
        second_promotion = trainer.stage_calls[16:18]
        second_anchor = trainer.stage_calls[18:20]
        self.assertEqual(len({call[2] for call in first_screen}), 1)
        self.assertEqual(len({call[2] for call in first_promotion}), 1)
        self.assertEqual(len({call[2] for call in second_screen}), 1)
        self.assertNotEqual(first_screen[0][2], second_screen[0][2])
        self.assertNotEqual(first_promotion[0][2], second_promotion[0][2])
        self.assertEqual(first_anchor[0][2], second_anchor[0][2])
        self.assertEqual(result["completed_generations"], 2)
        self.assertEqual(len(result["history"]), 2)
        self.assertIn("sigma_mean", result["history"][0])
        self.assertTrue(result["history"][0]["anchor_evaluated"])

    def test_warm_start_is_the_initial_distribution_center(self):
        warm_policy = MovementPolicy(
            [0.25] * MovementPolicy.genome_size,
            (5.0, 30.0),
            60.0,
        )
        trainer = self.trainer(initial_policy=warm_policy)
        trainer._initialize_distribution()

        population = trainer._sample_population()

        self.assertEqual(population[0].weights, warm_policy.weights)
        self.assertIsNot(population[0].weights, warm_policy.weights)

    def test_checkpoint_resume_matches_uninterrupted_search(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "optimizer.json"
            uninterrupted = self.trainer(
                generations=4,
                checkpoint_path=None,
            ).run()
            first_part = self.trainer(
                generations=2,
                checkpoint_path=checkpoint,
            ).run()
            self.assertEqual(first_part["completed_generations"], 2)
            self.assertTrue(checkpoint.exists())
            resumed = self.trainer(
                generations=4,
                checkpoint_path=checkpoint,
                resume_checkpoint=checkpoint,
            ).run()

        self.assertEqual(resumed["completed_generations"], 4)
        self.assertEqual(
            resumed["best"]["policy"].weights,
            uninterrupted["best"]["policy"].weights,
        )
        self.assertEqual(
            resumed["optimizer_state"]["mean"],
            uninterrupted["optimizer_state"]["mean"],
        )
        self.assertEqual(
            resumed["optimizer_state"]["sigma"],
            uninterrupted["optimizer_state"]["sigma"],
        )
        self.assertEqual(
            [entry["best_fitness"] for entry in resumed["history"]],
            [entry["best_fitness"] for entry in uninterrupted["history"]],
        )

    def test_robust_fitness_penalizes_scenario_variability(self):
        policy = MovementPolicy(
            [0.0] * MovementPolicy.genome_size,
            (5.0, 30.0),
            60.0,
        )
        candidate = _candidate_from_evaluation(
            policy,
            {
                "mean_fitness": 5.0,
                "evaluations": [{"fitness": 0.0}, {"fitness": 10.0}],
            },
            robustness_penalty=0.25,
            stage="anchor",
        )

        self.assertAlmostEqual(candidate["scenario_mean_fitness"], 5.0)
        self.assertAlmostEqual(candidate["scenario_fitness_std"], 5.0)
        self.assertAlmostEqual(candidate["fitness"], 3.75)

    def test_anchor_stagnation_reheats_once_and_can_stop_early(self):
        trainer = self.trainer(
            generations=5,
            anchor_interval=1,
            stagnation_patience=1,
            reheat_factor=1.5,
            early_stop_patience=2,
        )

        def flat_evaluation(policy, duration_s, scenario_pairs, stage):
            names = tuple(
                (profile.get("name", "unnamed"), seed)
                for profile, seed in scenario_pairs
            )
            return _candidate_from_evaluation(
                policy,
                {
                    "mean_fitness": 1.0,
                    "mean_metrics": {},
                    "evaluations": [
                        {"fitness": 1.0} for _ in scenario_pairs
                    ],
                    "requested_scenarios": names,
                    "evaluated_scenarios": names,
                },
                trainer.robustness_penalty,
                stage,
            )

        trainer._evaluate_policy = flat_evaluation
        result = trainer.run()

        self.assertTrue(result["stopped_early"])
        self.assertEqual(result["completed_generations"], 3)
        self.assertEqual(
            [item["sigma_reheated"] for item in result["history"]],
            [False, True, True],
        )
        self.assertAlmostEqual(result["history"][0]["sigma_mean"], 0.12)
        self.assertAlmostEqual(result["history"][1]["sigma_mean"], 0.18)
        self.assertAlmostEqual(result["history"][2]["sigma_mean"], 0.27)

    def test_parallel_es_matches_sequential_es(self):
        config = build_runtime_config(CONFIG)
        options = {
            "optimizer": "diagonal-es",
            "population_size": 2,
            "generations": 1,
            "seeds": (1,),
            "evaluation_duration_s": 0.1,
            "timestep_s": 1 / 30,
            "traffic_profiles": (),
            "random_seed": 23,
            "screening_duration_s": 0.05,
            "screening_scenarios": 1,
            "promotion_duration_s": 0.1,
            "promotion_scenarios": 1,
            "anchor_interval": 1,
            "anchor_candidates": 1,
            "checkpoint_path": None,
        }

        sequential = MovementPolicyEvolution(
            config,
            workers=1,
            **options,
        ).run()
        parallel = MovementPolicyEvolution(
            config,
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
