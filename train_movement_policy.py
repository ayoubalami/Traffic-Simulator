"""Train and save the independent-score movement policy."""

import argparse
import json
import multiprocessing
import os
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from config import CONFIG, build_runtime_config
from simulation import (
    MovementPolicyEvolution,
    evaluate_movement_policy_across_seeds,
)
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
            "Train independent vehicle-movement and pedestrian-WALK request "
            "scores with safe multi-hot decoding."
        ),
    )
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--seeds", type=parse_seeds, default=(1, 2, 3))
    parser.add_argument("--evaluation-duration", type=float, default=90.0)
    parser.add_argument(
        "--timestep",
        type=float,
        default=1 / 30,
        help="fixed physics timestep in seconds (default: 1/30)",
    )
    parser.add_argument(
        "--speed-factor",
        type=float,
        default=1.0,
        help=(
            "deprecated compatibility option; headless training always keeps "
            "the fixed --timestep physics"
        ),
    )
    parser.add_argument(
        "--optimizer",
        choices=("diagonal-es", "genetic"),
        default="diagonal-es",
        help="continuous diagonal evolution strategy or the legacy GA",
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
    parser.add_argument("--maximum-green", type=float, default=45.0)
    parser.add_argument("--random-seed", type=int, default=42)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument(
        "--warm-start",
        type=Path,
        default=None,
        help="initialize the search around an existing movement-policy JSON",
    )
    initialization.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="resume an optimizer checkpoint; --generations is the total target",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="optimizer checkpoint path (default: OUTPUT.checkpoint.json)",
    )
    parser.add_argument("--initial-sigma", type=float, default=0.12)
    parser.add_argument("--sigma-min", type=float, default=0.01)
    parser.add_argument("--sigma-max", type=float, default=0.75)
    parser.add_argument("--elite-fraction", type=float, default=0.20)
    parser.add_argument(
        "--distribution-learning-rate",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--stagnation-patience",
        type=int,
        default=3,
        help="anchor checks without improvement before increasing exploration",
    )
    parser.add_argument("--reheat-factor", type=float, default=1.5)
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=6,
        help="stop after this many anchor evaluations without improvement; 0 disables",
    )
    parser.add_argument(
        "--screen-duration",
        type=float,
        default=20.0,
        help="short first-stage duration for every candidate",
    )
    parser.add_argument(
        "--screen-scenarios",
        type=int,
        default=1,
        help="rotating common scenarios used by the first stage",
    )
    parser.add_argument("--promotion-fraction", type=float, default=0.25)
    parser.add_argument(
        "--promotion-duration",
        type=float,
        default=45.0,
        help="second-stage duration for promoted candidates",
    )
    parser.add_argument(
        "--promotion-scenarios",
        type=int,
        default=2,
        help="rotating common scenarios used by the second stage",
    )
    parser.add_argument(
        "--anchor-interval",
        type=int,
        default=5,
        help="evaluate finalists on the fixed anchor set every N generations",
    )
    parser.add_argument(
        "--anchor-candidates",
        type=int,
        default=2,
        help="number of promoted candidates checked on the fixed anchor set",
    )
    parser.add_argument(
        "--robustness-penalty",
        type=float,
        default=0.25,
        help="fitness = scenario mean - this value times scenario standard deviation",
    )
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
        default=Path("models/movement_policy_v3.json"),
    )
    return parser.parse_args()


def load_warm_start_policy(path, duration_bounds_s, max_red_duration_s):
    """Load only a schema-compatible movement policy for optimizer seeding."""
    if not path.exists():
        raise FileNotFoundError(f"warm-start policy not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    format_version = data.get("format_version")
    if (
        format_version
        not in (
            LEGACY_MOVEMENT_POLICY_FORMAT_VERSION,
            MOVEMENT_POLICY_FORMAT_VERSION,
        )
        or data.get("policy_type") != "movement_multi_hot"
        or tuple(data.get("movements", ())) != MOVEMENT_NAMES
    ):
        raise ValueError("warm-start file is not a compatible movement policy")
    network = data.get("network", {})
    if format_version == LEGACY_MOVEMENT_POLICY_FORMAT_VERSION:
        if (
            network.get("input_size")
            != len(LEGACY_MOVEMENT_INPUT_FEATURE_NAMES)
            or tuple(network.get("input_features", ()))
            != LEGACY_MOVEMENT_INPUT_FEATURE_NAMES
            or network.get("hidden_size") != MovementPolicy.hidden_size
            or network.get("output_size") != len(MOVEMENT_NAMES)
        ):
            raise ValueError(
                "warm-start policy has an incompatible network schema"
            )
        # Warm-starts seed the new 16-output search, so unlike deployment of
        # a legacy baseline they must enable the appended WALK neurons. This
        # keeps the evaluated champion identical to the saved format-3 model.
        return MovementPolicy(
            migrate_legacy_movement_policy_weights(
                data.get("weights", ())
            ),
            duration_bounds_s,
            max_red_duration_s,
        )
    if (
        tuple(data.get("pedestrian_outputs", ()))
        != PEDESTRIAN_OUTPUT_NAMES
        or tuple(data.get("outputs", ())) != POLICY_OUTPUT_NAMES
        or network.get("input_size") != MovementPolicy.input_size
        or tuple(network.get("input_features", ()))
        != MOVEMENT_INPUT_FEATURE_NAMES
        or network.get("hidden_size") != MovementPolicy.hidden_size
        or network.get("output_size") != MovementPolicy.output_size
        or tuple(network.get("output_names", ())) != POLICY_OUTPUT_NAMES
    ):
        raise ValueError("warm-start policy has an incompatible network schema")
    return MovementPolicy(
        data.get("weights", ()),
        duration_bounds_s,
        max_red_duration_s,
    )


def main():
    args = parse_arguments()
    if args.speed_factor <= 0:
        raise SystemExit("--speed-factor must be positive")
    if args.timestep <= 0:
        raise SystemExit("--timestep must be positive")
    if args.workers < 1:
        raise SystemExit("--workers must be a positive integer")
    minimum_population = 4 if args.optimizer == "diagonal-es" else 2
    if args.population < minimum_population:
        raise SystemExit(
            f"--population must be at least {minimum_population} for "
            f"{args.optimizer}"
        )
    if args.generations < 1:
        raise SystemExit("--generations must be positive")
    for option, value in (
        ("--evaluation-duration", args.evaluation_duration),
        ("--screen-duration", args.screen_duration),
        ("--promotion-duration", args.promotion_duration),
    ):
        if value <= 0:
            raise SystemExit(f"{option} must be positive")
    if not 0 < args.elite_fraction <= 1:
        raise SystemExit("--elite-fraction must be in (0, 1]")
    if not 0 < args.promotion_fraction <= 1:
        raise SystemExit("--promotion-fraction must be in (0, 1]")
    if not 0 < args.distribution_learning_rate <= 1:
        raise SystemExit("--distribution-learning-rate must be in (0, 1]")
    if args.initial_sigma <= 0 or args.sigma_min <= 0:
        raise SystemExit("sigma values must be positive")
    if args.sigma_max < args.sigma_min:
        raise SystemExit("--sigma-max must be at least --sigma-min")
    if args.screen_scenarios < 1 or args.promotion_scenarios < 1:
        raise SystemExit("stage scenario counts must be positive")
    if args.anchor_interval < 1 or args.anchor_candidates < 1:
        raise SystemExit("anchor settings must be positive")
    if args.stagnation_patience < 1 or args.early_stop_patience < 0:
        raise SystemExit("patience values must be non-negative")
    if args.reheat_factor < 1:
        raise SystemExit("--reheat-factor must be at least 1")
    if args.robustness_penalty < 0:
        raise SystemExit("--robustness-penalty cannot be negative")
    validation_duration = (
        args.evaluation_duration
        if args.validation_duration is None
        else args.validation_duration
    )
    if validation_duration <= 0:
        raise SystemExit("--validation-duration must be positive")
    duration_bounds_s = (args.minimum_green, args.maximum_green)

    def log_generation(progress):
        sigma = progress.get("sigma_mean")
        optimizer_details = (
            f" | sigma={sigma:.4f} | stagnant={progress.get('stagnation', 0)}"
            if sigma is not None
            else ""
        )
        anchor = " | anchor=yes" if progress.get("anchor_evaluated") else ""
        print(
            f"Generation {progress['generation_number']}/{args.generations} | "
            f"generation={progress['generation_time_s']:.2f}s | "
            f"elapsed={progress['elapsed_time_s']:.2f}s | "
            f"best={progress['best_fitness']:.2f} | "
            f"mean={progress['mean_fitness']:.2f} | "
            f"global_best={progress['global_best_fitness']:.2f}"
            f"{optimizer_details}{anchor}",
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

    initial_policy = None
    if args.warm_start is not None:
        initial_policy = load_warm_start_policy(
            args.warm_start,
            duration_bounds_s,
            timing.get("max_red_duration_s", 60.0),
        )
    checkpoint_path = args.checkpoint or args.resume
    if checkpoint_path is None:
        checkpoint_path = Path(f"{args.output}.checkpoint.json")

    trainer = MovementPolicyEvolution(
        runtime_config,
        duration_bounds_s=duration_bounds_s,
        population_size=args.population,
        generations=args.generations,
        optimizer=args.optimizer,
        seeds=args.seeds,
        evaluation_duration_s=args.evaluation_duration,
        timestep_s=args.timestep,
        speed_factor=args.speed_factor,
        traffic_profiles=traffic_profiles,
        workers=args.workers,
        progress_callback=log_generation,
        random_seed=args.random_seed,
        initial_policy=initial_policy,
        initial_sigma=args.initial_sigma,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        elite_fraction=args.elite_fraction,
        distribution_learning_rate=args.distribution_learning_rate,
        promotion_fraction=args.promotion_fraction,
        screening_duration_s=args.screen_duration,
        screening_scenarios=args.screen_scenarios,
        promotion_duration_s=args.promotion_duration,
        promotion_scenarios=args.promotion_scenarios,
        anchor_interval=args.anchor_interval,
        anchor_candidates=args.anchor_candidates,
        robustness_penalty=args.robustness_penalty,
        checkpoint_path=checkpoint_path,
        resume_checkpoint=args.resume,
        stagnation_patience=args.stagnation_patience,
        reheat_factor=args.reheat_factor,
        early_stop_patience=(
            None if args.early_stop_patience == 0 else args.early_stop_patience
        ),
    )
    if args.speed_factor != 1.0:
        print(
            "Note: --speed-factor no longer changes headless physics; "
            "parallel workers provide acceleration.",
            flush=True,
        )
    print(
        "Training movement neural policy "
        f"({args.population} candidates, {args.generations} generations, "
        f"{len(traffic_profiles)} profiles, seeds={args.seeds}, "
        f"optimizer={args.optimizer}, dt={args.timestep:g}s, "
        f"workers={trainer.workers})..."
    )
    result = trainer.run()
    best = result["best"]
    optimizer_state = result.get("optimizer_state") or {}
    optimizer_sigma = optimizer_state.get("sigma") or ()
    optimizer_summary = {
        "sigma_min": min(optimizer_sigma) if optimizer_sigma else None,
        "sigma_mean": (
            sum(optimizer_sigma) / len(optimizer_sigma)
            if optimizer_sigma
            else None
        ),
        "sigma_max": max(optimizer_sigma) if optimizer_sigma else None,
        "anchor_stagnation": optimizer_state.get("anchor_stagnation"),
    }
    validation = None
    if not args.skip_validation:
        print(
            "Validating best policy with the training physics timestep on holdout seeds "
            f"{args.validation_seeds}...",
            flush=True,
        )
        validation = evaluate_movement_policy_across_seeds(
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
        "fitness_version": 4,
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
            "output_activation": "sigmoid",
        },
        "decoder": dict(runtime_config.get("movement_controller", {})),
        "pedestrian_decoder": dict(
            runtime_config.get("pedestrian_signals", {})
        ),
        "duration_bounds_s": duration_bounds_s,
        "max_red_duration_s": timing.get("max_red_duration_s", 60.0),
        "weights": best["policy"].weights,
        "fitness": best["fitness"],
        "mean_metrics": best["mean_metrics"],
        "scenario_evaluations": best["scenario_evaluations"],
        "validation": validation,
        "history": result["history"],
        "training": {
            "population_size": args.population,
            "generations": args.generations,
            "optimizer": args.optimizer,
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
            "warm_start": str(args.warm_start) if args.warm_start else None,
            "checkpoint": str(checkpoint_path),
            "resume": str(args.resume) if args.resume else None,
            "initial_sigma": args.initial_sigma,
            "sigma_min": args.sigma_min,
            "sigma_max": args.sigma_max,
            "elite_fraction": args.elite_fraction,
            "distribution_learning_rate": args.distribution_learning_rate,
            "stagnation_patience": args.stagnation_patience,
            "reheat_factor": args.reheat_factor,
            "early_stop_patience": args.early_stop_patience,
            "screening_duration_s": args.screen_duration,
            "screening_scenarios": args.screen_scenarios,
            "promotion_fraction": args.promotion_fraction,
            "promotion_duration_s": args.promotion_duration,
            "promotion_scenarios": args.promotion_scenarios,
            "anchor_interval": args.anchor_interval,
            "anchor_candidates": args.anchor_candidates,
            "robustness_penalty": args.robustness_penalty,
            "completed_generations": len(result["history"]),
            "stopped_early": result.get("stopped_early", False),
            # Full distribution vectors live in the resumable checkpoint;
            # deployable model JSON only needs a compact diagnostic summary.
            "optimizer_summary": optimizer_summary,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Best fitness: {best['fitness']:.2f}")
    print(f"Total training time: {result['training_time_s']:.2f}s")
    print(f"Saved movement policy to: {args.output}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
