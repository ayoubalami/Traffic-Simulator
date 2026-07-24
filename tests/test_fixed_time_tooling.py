import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import compare_policies
import main_fixed_time
from simulation import MovementTrafficLightController, load_fixed_time_plan


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "models" / "fixed_time_policy_v1.json"


class FixedTimeToolingTests(unittest.TestCase):
    def test_default_plan_is_safe_conventional_six_stage_baseline(self):
        plan = load_fixed_time_plan(DEFAULT_PLAN)

        self.assertEqual(plan.control_scope, "vehicles_only")
        self.assertEqual(len(plan.stages), 6)
        self.assertAlmostEqual(
            sum(stage.duration_s for stage in plan.stages),
            72.0,
        )
        self.assertEqual(
            tuple(stage.duration_s for stage in plan.stages),
            (8.0, 20.0, 8.0, 8.0, 20.0, 8.0),
        )
        self.assertTrue(
            all(
                MovementTrafficLightController.is_conflict_free(
                    stage.movements
                )
                for stage in plan.stages
            )
        )

    def test_fixed_runner_defaults_to_publication_baseline(self):
        with patch.object(sys, "argv", ["main_fixed_time.py"]):
            args = main_fixed_time.parse_arguments()

        self.assertEqual(args.plan, main_fixed_time.PLAN_PATH)

    def test_compare_defaults_include_fixed_and_both_vehicle_models(self):
        with patch.object(sys, "argv", ["compare_policies.py"]):
            args = compare_policies.parse_arguments()

        self.assertEqual(
            args.fixed_plan,
            Path("models/fixed_time_policy_v1.json"),
        )
        self.assertEqual(
            args.movement_model,
            Path("models/vehicle_movement_policy_v7.json"),
        )
        self.assertEqual(
            args.uncertain_movement_model,
            Path("models/vehicle_movement_policy_v8.json"),
        )

    def test_comparison_runs_all_four_on_full_scenario_matrix_and_saves_json(self):
        config = {
            "road_users": {"pedestrians_enabled": False},
            "six_phase_fitness": {
                "abort_remaining_seeds_on_gridlock": True,
            },
        }
        fixed = SimpleNamespace(control_scope="vehicles_only")
        movement = SimpleNamespace(control_scope="vehicles_only")
        uncertain_movement = SimpleNamespace(control_scope="vehicles_only")
        categorical = object()
        result = {
            "mean_fitness": 1.0,
            "mean_metrics": {},
            "requested_scenario_count": 2,
            "evaluated_scenario_count": 2,
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "comparison.json"
            arguments = SimpleNamespace(
                fixed_plan=DEFAULT_PLAN,
                categorical_model=Path("categorical.json"),
                movement_model=Path("movement.json"),
                uncertain_movement_model=Path("uncertain.json"),
                seeds=(1, 2),
                evaluation_duration=10.0,
                speed_factor=1.0,
                timestep=0.1,
                json_output=output_path,
            )
            fixed_evaluator = MagicMock(return_value=result)
            categorical_evaluator = MagicMock(return_value=result)
            movement_evaluator = MagicMock(return_value=result)
            with (
                patch.object(compare_policies, "parse_arguments", return_value=arguments),
                patch.object(compare_policies, "build_runtime_config", return_value=config),
                patch.object(compare_policies, "load_fixed_time_plan", return_value=fixed),
                patch.object(compare_policies, "load_six_phase_policy", return_value=categorical),
                patch.object(
                    compare_policies,
                    "load_movement_policy",
                    side_effect=(movement, uncertain_movement),
                ),
                patch.object(
                    compare_policies,
                    "evaluate_fixed_time_policy_across_seeds",
                    fixed_evaluator,
                ),
                patch.object(
                    compare_policies,
                    "evaluate_six_phase_policy_across_seeds",
                    categorical_evaluator,
                ),
                patch.object(
                    compare_policies,
                    "evaluate_movement_policy_across_seeds",
                    movement_evaluator,
                ),
                redirect_stdout(StringIO()),
            ):
                compare_policies.main()

            self.assertFalse(
                config["six_phase_fitness"]
                ["abort_remaining_seeds_on_gridlock"]
            )
            for evaluator in (fixed_evaluator, categorical_evaluator):
                evaluator.assert_called_once()
                self.assertIs(evaluator.call_args.args[0], config)
            self.assertEqual(movement_evaluator.call_count, 2)
            for call in movement_evaluator.call_args_list:
                self.assertIs(call.args[0], config)

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertIn("fixed", payload)
            self.assertIn("categorical", payload)
            self.assertIn("movement", payload)
            self.assertIn("uncertain_movement", payload)
            self.assertEqual(
                payload["models"]["uncertain_movement"],
                "uncertain.json",
            )


if __name__ == "__main__":
    unittest.main()
