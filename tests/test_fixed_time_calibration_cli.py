import argparse
import unittest
from unittest.mock import patch

from calibrate_fixed_time_plan import (
    _evaluate_candidate,
    build_calibrated_output_plan,
    calibrate_plan,
    clone_plan_with_durations,
    parse_arguments,
    score_evaluation,
)
from simulation.fixed_time import FixedTimeMovementPlan, FixedTimeStage


def _plan():
    return FixedTimeMovementPlan(
        name="paper_baseline",
        stages=(
            FixedTimeStage(
                name="north",
                duration_s=10.0,
                movements=("north_through",),
            ),
            FixedTimeStage(
                name="east",
                duration_s=10.0,
                movements=("east_through",),
            ),
        ),
        metadata={"source": "equal_split"},
    )


class FixedTimeCalibrationCliTests(unittest.TestCase):
    def test_argument_parsing_exposes_reproducible_search_controls(self):
        args = parse_arguments(
            [
                "--plan",
                "models/source.json",
                "--output",
                "models/calibrated.json",
                "--population",
                "9",
                "--generations",
                "4",
                "--min-green",
                "6",
                "--max-green",
                "30",
                "--mutation-sigma",
                "2.5",
                "--seeds",
                "4,8,15",
                "--evaluation-duration",
                "120",
                "--timestep",
                "0.05",
                "--workers",
                "1",
                "--random-seed",
                "99",
                "--robustness-penalty",
                "0.4",
            ]
        )

        self.assertEqual(args.population, 9)
        self.assertEqual(args.generations, 4)
        self.assertEqual(args.seeds, (4, 8, 15))
        self.assertAlmostEqual(args.min_green, 6.0)
        self.assertAlmostEqual(args.max_green, 30.0)
        self.assertAlmostEqual(args.mutation_sigma, 2.5)
        self.assertEqual(args.random_seed, 99)
        self.assertAlmostEqual(args.robustness_penalty, 0.4)

    def test_clone_changes_only_stage_durations(self):
        source = _plan()
        clone = clone_plan_with_durations(source, (12.5, 7.0))

        self.assertEqual(
            tuple(stage.duration_s for stage in clone.stages),
            (12.5, 7.0),
        )
        self.assertEqual(
            tuple(stage.movements for stage in clone.stages),
            tuple(stage.movements for stage in source.stages),
        )
        self.assertEqual(
            tuple(stage.name for stage in clone.stages),
            tuple(stage.name for stage in source.stages),
        )
        self.assertEqual(
            tuple(stage.duration_s for stage in source.stages),
            (10.0, 10.0),
        )
        self.assertEqual(clone.metadata, source.metadata)

    def test_robust_score_uses_scenario_population_standard_deviation(self):
        summary = score_evaluation(
            {
                "evaluations": (
                    {"fitness": 10.0},
                    {"fitness": 14.0},
                )
            },
            robustness_penalty=0.5,
        )

        self.assertAlmostEqual(summary["scenario_mean_fitness"], 12.0)
        self.assertAlmostEqual(summary["scenario_fitness_std"], 2.0)
        self.assertAlmostEqual(summary["robust_fitness"], 11.0)
        self.assertEqual(summary["scenario_count"], 2)

    def test_candidate_evaluation_uses_training_matrix_and_disables_abort(self):
        source = _plan()
        profiles = ({"name": "balanced"}, {"name": "east_rush"})
        evaluation = {
            "evaluations": ({"fitness": 3.0}, {"fitness": 5.0}),
            "mean_fitness": 4.0,
            "skipped_scenario_count": 0,
        }
        payload = {
            "config": {"simulation": {}},
            "plan": source,
            "durations_s": (12.0, 8.0),
            "seeds": (1, 2),
            "traffic_profiles": profiles,
            "evaluation_duration_s": 60.0,
            "timestep_s": 0.1,
            "robustness_penalty": 0.25,
        }

        with patch(
            "calibrate_fixed_time_plan.evaluate_fixed_time_policy_across_seeds",
            return_value=evaluation,
        ) as evaluate:
            result = _evaluate_candidate(payload)

        evaluated_plan = evaluate.call_args.args[1]
        self.assertEqual(
            tuple(stage.duration_s for stage in evaluated_plan.stages),
            (12.0, 8.0),
        )
        self.assertEqual(evaluate.call_args.kwargs["seeds"], (1, 2))
        self.assertEqual(
            evaluate.call_args.kwargs["traffic_profiles"],
            profiles,
        )
        self.assertFalse(
            evaluate.call_args.kwargs[
                "abort_remaining_seeds_on_gridlock"
            ]
        )
        self.assertAlmostEqual(result["robust_fitness"], 3.75)

    def test_calibration_is_reproducible_and_preserves_movements(self):
        source = _plan()

        def fake_candidate(payload):
            durations = tuple(payload["durations_s"])
            fitness = -sum(
                (value - target) ** 2
                for value, target in zip(durations, (12.0, 8.0))
            )
            return {
                "durations_s": durations,
                "scenario_mean_fitness": fitness,
                "scenario_fitness_std": 0.0,
                "robust_fitness": fitness,
                "scenario_count": 1,
            }

        kwargs = {
            "config": {},
            "plan": source,
            "traffic_profiles": ({"name": "training"},),
            "seeds": (1,),
            "population_size": 5,
            "generations": 2,
            "minimum_green_s": 5.0,
            "maximum_green_s": 20.0,
            "mutation_sigma_s": 2.0,
            "workers": 1,
            "random_seed": 7,
        }
        with patch(
            "calibrate_fixed_time_plan._evaluate_candidate",
            side_effect=fake_candidate,
        ):
            first = calibrate_plan(**kwargs)
        with patch(
            "calibrate_fixed_time_plan._evaluate_candidate",
            side_effect=fake_candidate,
        ):
            second = calibrate_plan(**kwargs)

        self.assertEqual(first["durations_s"], second["durations_s"])
        self.assertGreater(first["robust_fitness"], -8.0)
        self.assertEqual(
            tuple(stage.movements for stage in first["plan"].stages),
            tuple(stage.movements for stage in source.stages),
        )

    def test_output_metadata_records_training_and_leaves_holdout_untouched(self):
        source = _plan()
        result = {
            "plan": clone_plan_with_durations(source, (12.0, 8.0)),
            "scenario_mean_fitness": 100.0,
            "scenario_fitness_std": 4.0,
            "robust_fitness": 99.0,
            "scenario_count": 6,
        }
        args = argparse.Namespace(
            seeds=(1, 2, 3),
            evaluation_duration=90.0,
            timestep=1 / 30,
            population=13,
            generations=10,
            min_green=5.0,
            max_green=45.0,
            mutation_sigma=3.0,
            workers=1,
            random_seed=42,
            robustness_penalty=0.25,
        )
        profiles = ({"name": "balanced"}, {"name": "rush"})
        output = build_calibrated_output_plan(
            source,
            result,
            {
                "fitness": {"throughput_rate_reward": 10000.0},
                "six_phase_fitness": {"gridlock_penalty": 100000.0},
            },
            args,
            profiles,
        )

        payload = output.to_dict()
        calibration = payload["metadata"]["calibration"]
        self.assertEqual(payload["format_version"], 1)
        self.assertEqual(calibration["training_seeds"], [1, 2, 3])
        self.assertEqual(calibration["training_profiles"], list(profiles))
        self.assertEqual(calibration["scenario_count"], 6)
        self.assertFalse(calibration["holdout_evaluated"])
        self.assertTrue(calibration["holdout_untouched"])
        self.assertEqual(payload["metadata"]["source"], "equal_split")


if __name__ == "__main__":
    unittest.main()
