"""Compare categorical and movement-level policies on identical scenarios."""

import argparse
import json
from pathlib import Path

from config import CONFIG, build_runtime_config
from main_movement_policy import load_movement_policy
from main_six_phase import load_six_phase_policy
from simulation import (
    evaluate_movement_policy_across_seeds,
    evaluate_six_phase_policy_across_seeds,
)


METRICS = (
    "throughput",
    "throughput_rate",
    "avg_vehicle_wait_time_all",
    "stops_per_vehicle",
    "max_avg_pre_intersection_wait_time",
    "avg_pedestrian_wait_time_all",
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
        description="Compare both policy encodings using identical seeds and profiles."
    )
    parser.add_argument(
        "--categorical-model",
        type=Path,
        default=Path("models/six_phase_policy_v8.json"),
    )
    parser.add_argument(
        "--movement-model",
        type=Path,
        default=Path("models/movement_policy_v1.json"),
    )
    parser.add_argument("--seeds", type=parse_seeds, default=(1, 2, 3))
    parser.add_argument("--evaluation-duration", type=float, default=300.0)
    parser.add_argument(
        "--speed-factor",
        type=float,
        default=1.0,
        help="use 1 for final scientific comparison",
    )
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_arguments()
    config = build_runtime_config(CONFIG)
    categorical = load_six_phase_policy(args.categorical_model)
    movement = load_movement_policy(args.movement_model)
    options = {
        "seeds": args.seeds,
        "duration_s": args.evaluation_duration,
        "speed_factor": args.speed_factor,
    }
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
    metric_width = max(38, max(len(metric) for metric in METRICS))
    print(
        f"{'Metric':{metric_width}} "
        f"{'Categorical':>14} {'Movement':>14} {'Delta':>14}"
    )
    print("-" * (metric_width + 46))
    print(
        f"{'fitness':{metric_width}} "
        f"{categorical_result['mean_fitness']:14.2f} "
        f"{movement_result['mean_fitness']:14.2f} "
        f"{movement_result['mean_fitness'] - categorical_result['mean_fitness']:14.2f}"
    )
    for metric in METRICS:
        categorical_value = float(
            categorical_result["mean_metrics"].get(metric, 0.0)
        )
        movement_value = float(movement_result["mean_metrics"].get(metric, 0.0))
        print(
            f"{metric:{metric_width}} {categorical_value:14.2f} "
            f"{movement_value:14.2f} "
            f"{movement_value - categorical_value:14.2f}"
        )
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(
                {
                    "categorical": categorical_result,
                    "movement": movement_result,
                    "seeds": args.seeds,
                    "evaluation_duration_s": args.evaluation_duration,
                    "speed_factor": args.speed_factor,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Saved comparison to: {args.json_output}")


if __name__ == "__main__":
    main()
