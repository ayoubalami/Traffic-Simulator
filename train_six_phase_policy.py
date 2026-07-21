"""Train and save the separate paired/individual-approach neural policy."""

import argparse
import json
import multiprocessing
import os
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from config import CONFIG, build_runtime_config
from simulation import (
    SixPhasePolicyEvolution,
    evaluate_six_phase_policy_across_seeds,
)
from simulation.six_phase_neuroevolution import (
    INPUT_FEATURE_NAMES,
    PHASE_NAMES,
    SIX_PHASE_POLICY_FORMAT_VERSION,
    SixPhasePolicy,
)


def parse_seeds(value):
    seeds = tuple(int(seed.strip()) for seed in value.split(",") if seed.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("provide at least one comma-separated seed")
    return seeds


def parse_profile_names(value):
    names = tuple(name.strip() for name in value.split(",") if name.strip())
    if not names:
        raise argparse.ArgumentTypeError("provide at least one profile name")
    return names


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Train main phases with independently actuated safe right arrows."
        ),
    )
    parser.add_argument("--population", type=int, default=50)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--seeds", type=parse_seeds, default=(1,))
    parser.add_argument("--evaluation-duration", type=float, default=60.0)
    parser.add_argument(
        "--timestep",
        type=float,
        default=1 / 30,
        help="fixed physics timestep in seconds",
    )
    parser.add_argument(
        "--speed-factor",
        type=float,
        default=1.0,
        help="deprecated compatibility option; it no longer changes physics",
    )
    parser.add_argument(
        "--validation-seeds",
        type=parse_seeds,
        default=(101, 102, 103),
        help="holdout seeds evaluated after training",
    )
    parser.add_argument(
        "--validation-duration",
        type=float,
        default=None,
        help="holdout duration; default: use --evaluation-duration",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="skip the final fixed-timestep holdout evaluation",
    )
    parser.add_argument("--minimum-green", type=float, default=10.0)
    parser.add_argument("--maximum-green", type=float, default=30.0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 1) - 1)),
        help="candidate evaluation processes; use 1 to disable parallelism",
    )
    parser.add_argument(
        "--profiles",
        type=parse_profile_names,
        default=None,
        help="comma-separated configured profile names; default: all",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/six_phase_policy_v9.json"),
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.speed_factor <= 0:
        raise SystemExit("--speed-factor must be positive")
    if args.timestep <= 0:
        raise SystemExit("--timestep must be positive")
    if args.workers < 1:
        raise SystemExit("--workers must be a positive integer")
    validation_duration = (
        args.evaluation_duration
        if args.validation_duration is None
        else args.validation_duration
    )
    if validation_duration <= 0:
        raise SystemExit("--validation-duration must be positive")
    duration_bounds_s = (args.minimum_green, args.maximum_green)

    def log_generation(progress):
        print(
            f"Generation {progress['generation_number']}/{args.generations} | "
            f"generation={progress['generation_time_s']:.2f}s | "
            f"elapsed={progress['elapsed_time_s']:.2f}s | "
            f"best={progress['best_fitness']:.2f} | "
            f"mean={progress['mean_fitness']:.2f} | "
            f"global_best={progress['global_best_fitness']:.2f}",
            flush=True,
        )

    runtime_config = build_runtime_config(CONFIG)
    timing = runtime_config["traffic_lights"]
    timing["min_green_duration_s"] = args.minimum_green
    timing["max_green_duration_s"] = args.maximum_green
    configured_profiles = runtime_config.get("six_phase_training", {}).get(
        "traffic_profiles",
        [],
    )
    if args.profiles is None:
        traffic_profiles = configured_profiles
    else:
        profiles_by_name = {
            profile["name"]: profile for profile in configured_profiles
        }
        unknown = [name for name in args.profiles if name not in profiles_by_name]
        if unknown:
            raise SystemExit(f"unknown traffic profiles: {', '.join(unknown)}")
        traffic_profiles = [profiles_by_name[name] for name in args.profiles]
    trainer = SixPhasePolicyEvolution(
        runtime_config,
        duration_bounds_s=duration_bounds_s,
        population_size=args.population,
        generations=args.generations,
        seeds=args.seeds,
        evaluation_duration_s=args.evaluation_duration,
        timestep_s=args.timestep,
        speed_factor=args.speed_factor,
        traffic_profiles=traffic_profiles,
        workers=args.workers,
        progress_callback=log_generation,
        random_seed=args.random_seed,
    )
    print(
        "Training six-phase neural policy "
        f"({args.population} candidates, {args.generations} generations, "
        f"{len(traffic_profiles)} profiles, seeds={args.seeds}, "
        f"dt={args.timestep:g}s, workers={trainer.workers})..."
    )
    result = trainer.run()
    best = result["best"]
    validation = None
    if not args.skip_validation:
        print(
            "Validating best policy with the fixed physics timestep on holdout seeds "
            f"{args.validation_seeds}...",
            flush=True,
        )
        validation = evaluate_six_phase_policy_across_seeds(
            runtime_config,
            best["policy"],
            seeds=args.validation_seeds,
            duration_s=validation_duration,
            timestep_s=args.timestep,
            speed_factor=1.0,
            traffic_profiles=traffic_profiles,
        )
        print(
            f"Holdout fitness: {validation['mean_fitness']:.2f}",
            flush=True,
        )
    output = {
        "fitness_version": 5,
        "format_version": SIX_PHASE_POLICY_FORMAT_VERSION,
        "policy_type": "six_phase",
        "phases": list(PHASE_NAMES),
        "network": {
            "input_size": SixPhasePolicy.input_size,
            "input_features": list(INPUT_FEATURE_NAMES),
            "hidden_size": SixPhasePolicy.hidden_size,
            "output_size": SixPhasePolicy.output_size,
        },
        "duration_bounds_s": duration_bounds_s,
        "max_red_duration_s": runtime_config["traffic_lights"].get(
            "max_red_duration_s", 60.0
        ),
        "weights": best["policy"].weights,
        "fitness": best["fitness"],
        "mean_metrics": best["mean_metrics"],
        "scenario_evaluations": best["scenario_evaluations"],
        "validation": validation,
        "history": result["history"],
        "training": {
            "population_size": args.population,
            "generations": args.generations,
            "seeds": args.seeds,
            "evaluation_duration_s": args.evaluation_duration,
            "physics_timestep_s": args.timestep,
            "speed_factor_compatibility_value": args.speed_factor,
            "validation_seeds": args.validation_seeds,
            "validation_duration_s": validation_duration,
            "workers": trainer.workers,
            "training_time_s": result["training_time_s"],
            "fitness_weights": dict(CONFIG.get("fitness", {})),
            "six_phase_fitness_weights": dict(
                CONFIG.get("six_phase_fitness", {})
            ),
            "traffic_profiles": traffic_profiles,
            "random_seed": args.random_seed,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Best fitness: {best['fitness']:.2f}")
    print(f"Total training time: {result['training_time_s']:.2f}s")
    print(f"Saved six-phase policy to: {args.output}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
