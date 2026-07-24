"""Compare fixed and neural policies on identical traffic scenarios."""

import argparse
import json
from pathlib import Path

from config import CONFIG, apply_movement_control_scope, build_runtime_config
from main_movement_policy import load_movement_policy
from main_six_phase import load_six_phase_policy
from simulation import (
    evaluate_fixed_time_policy_across_seeds,
    evaluate_movement_policy_across_seeds,
    evaluate_six_phase_policy_across_seeds,
    load_fixed_time_plan,
)


METRICS = (
    "throughput",
    "throughput_rate",
    "pending_arrivals",
    "dropped_arrivals",
    "avg_vehicle_wait_time_all",
    "avg_system_wait_time_all",
    "stops_per_vehicle",
    "max_avg_pre_intersection_wait_time",
    "emergency_vehicle_completion_rate",
    "avg_emergency_vehicle_wait_time_all",
    "max_emergency_vehicle_wait_time",
    "avg_pedestrian_wait_time_all",
    "pedestrian_wait_time_p95",
    "pedestrian_completion_rate",
    "wasted_pedestrian_walk_fraction",
    "vehicle_pedestrian_crosswalk_cooccupancy_fraction",
    "vehicle_pedestrian_crosswalk_conflict_events",
    "hard_braking_vehicle_rate",
    "avg_excess_braking_intensity_per_vehicle",
    "phase_switches",
    "movement_set_changes",
    "transition_clearance_fraction",
    "green_movement_utilization",
    "wasted_green_movement_fraction",
    "intersection_blocking_rate",
    "avg_left_turn_delay",
    "avg_right_turn_delay",
    "gridlock_detected",
)


def parse_seeds(value):
    seeds = tuple(int(seed.strip()) for seed in value.split(",") if seed.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("provide at least one comma-separated seed")
    return seeds


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Compare fixed-time, categorical, standard movement, and "
            "uncertainty-trained movement controllers using identical seeds "
            "and traffic profiles."
        )
    )
    parser.add_argument(
        "--fixed-plan",
        type=Path,
        default=Path("models/fixed_time_policy_v1.json"),
    )
    parser.add_argument(
        "--categorical-model",
        type=Path,
        default=Path("models/six_phase_policy_v8.json"),
    )
    parser.add_argument(
        "--movement-model",
        type=Path,
        default=Path("models/vehicle_movement_policy_v7.json"),
    )
    parser.add_argument(
        "--uncertain-movement-model",
        type=Path,
        default=Path("models/vehicle_movement_policy_v8.json"),
        help=(
            "movement policy trained with camera uncertainty; it is evaluated "
            "under the same runtime camera settings as the standard model"
        ),
    )
    parser.add_argument("--seeds", type=parse_seeds, default=(1, 2, 3))
    parser.add_argument("--evaluation-duration", type=float, default=300.0)
    parser.add_argument(
        "--speed-factor",
        type=float,
        default=1.0,
        help="deprecated compatibility option; it no longer changes physics",
    )
    parser.add_argument("--timestep", type=float, default=1 / 30)
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.timestep <= 0:
        raise SystemExit("--timestep must be positive")
    if args.speed_factor <= 0:
        raise SystemExit("--speed-factor must be positive")
    config = build_runtime_config(CONFIG)
    fixed = load_fixed_time_plan(args.fixed_plan)
    categorical = load_six_phase_policy(args.categorical_model)
    movement = load_movement_policy(args.movement_model)
    uncertain_movement = load_movement_policy(args.uncertain_movement_model)
    fixed_scope = getattr(fixed, "control_scope", "vehicles_only")
    movement_scope = getattr(movement, "control_scope", None)
    uncertain_scope = getattr(uncertain_movement, "control_scope", None)
    incompatible_scopes = tuple(
        scope
        for scope in (movement_scope, uncertain_scope)
        if scope is not None and scope != fixed_scope
    )
    if incompatible_scopes:
        raise SystemExit(
            "The fixed and movement policies must use the same control scope "
            f"for a comparable experiment (fixed={fixed_scope}, "
            f"movement={movement_scope}, uncertain={uncertain_scope})."
        )
    apply_movement_control_scope(config, fixed_scope)
    # A failed/gridlocked controller must not shorten its scenario matrix.
    # Every method is therefore measured on every requested profile x seed.
    config.setdefault("six_phase_fitness", {})[
        "abort_remaining_seeds_on_gridlock"
    ] = False
    options = {
        "seeds": args.seeds,
        "duration_s": args.evaluation_duration,
        "timestep_s": args.timestep,
        "speed_factor": args.speed_factor,
    }
    fixed_result = evaluate_fixed_time_policy_across_seeds(
        config,
        fixed,
        **options,
    )
    categorical_result = evaluate_six_phase_policy_across_seeds(
        config,
        categorical,
        **options,
    )
    movement_result = evaluate_movement_policy_across_seeds(
        config,
        movement,
        **options,
    )
    uncertain_result = evaluate_movement_policy_across_seeds(
        config,
        uncertain_movement,
        **options,
    )
    metric_width = max(38, max(len(metric) for metric in METRICS))
    print(
        f"{'Metric':{metric_width}} "
        f"{'Fixed':>14} {'Categorical':>14} {'Movement':>14} "
        f"{'Uncertain':>14} {'Cat-Fixed':>14} {'Move-Fixed':>14} "
        f"{'Unc-Fixed':>14} {'Unc-Move':>14}"
    )
    print("-" * (metric_width + 121))
    print(
        f"{'fitness':{metric_width}} "
        f"{fixed_result['mean_fitness']:14.2f} "
        f"{categorical_result['mean_fitness']:14.2f} "
        f"{movement_result['mean_fitness']:14.2f} "
        f"{uncertain_result['mean_fitness']:14.2f} "
        f"{categorical_result['mean_fitness'] - fixed_result['mean_fitness']:14.2f} "
        f"{movement_result['mean_fitness'] - fixed_result['mean_fitness']:14.2f} "
        f"{uncertain_result['mean_fitness'] - fixed_result['mean_fitness']:14.2f} "
        f"{uncertain_result['mean_fitness'] - movement_result['mean_fitness']:14.2f}"
    )
    for metric in METRICS:
        fixed_value = float(fixed_result["mean_metrics"].get(metric, 0.0))
        categorical_value = float(
            categorical_result["mean_metrics"].get(metric, 0.0)
        )
        movement_value = float(movement_result["mean_metrics"].get(metric, 0.0))
        uncertain_value = float(
            uncertain_result["mean_metrics"].get(metric, 0.0)
        )
        print(
            f"{metric:{metric_width}} {fixed_value:14.2f} "
            f"{categorical_value:14.2f} "
            f"{movement_value:14.2f} "
            f"{uncertain_value:14.2f} "
            f"{categorical_value - fixed_value:14.2f} "
            f"{movement_value - fixed_value:14.2f} "
            f"{uncertain_value - fixed_value:14.2f} "
            f"{uncertain_value - movement_value:14.2f}"
        )
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(
                {
                    "fixed": fixed_result,
                    "categorical": categorical_result,
                    "movement": movement_result,
                    "uncertain_movement": uncertain_result,
                    "seeds": args.seeds,
                    "evaluation_duration_s": args.evaluation_duration,
                    "physics_timestep_s": args.timestep,
                    "speed_factor_compatibility_value": args.speed_factor,
                    "models": {
                        "fixed": str(args.fixed_plan),
                        "categorical": str(args.categorical_model),
                        "movement": str(args.movement_model),
                        "uncertain_movement": str(
                            args.uncertain_movement_model
                        ),
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Saved comparison to: {args.json_output}")


if __name__ == "__main__":
    main()
