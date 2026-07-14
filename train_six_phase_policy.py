"""Train and save the separate paired/individual-approach neural policy."""

import argparse
import json
from pathlib import Path

from config import CONFIG, build_runtime_config
from simulation import SixPhasePolicyEvolution
from simulation.six_phase_neuroevolution import PHASE_NAMES, SixPhasePolicy


def parse_seeds(value):
    seeds = tuple(int(seed.strip()) for seed in value.split(",") if seed.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("provide at least one comma-separated seed")
    return seeds


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train a six-phase policy with paired and individual greens.",
    )
    parser.add_argument("--population", type=int, default=50)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--seeds", type=parse_seeds, default=(1,))
    parser.add_argument("--evaluation-duration", type=float, default=60.0)
    parser.add_argument("--speed-factor", type=float, default=10.0)
    parser.add_argument("--minimum-green", type=float, default=10.0)
    parser.add_argument("--maximum-green", type=float, default=30.0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/six_phase_policy.json"),
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.speed_factor <= 0:
        raise SystemExit("--speed-factor must be positive")
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
    trainer = SixPhasePolicyEvolution(
        runtime_config,
        duration_bounds_s=duration_bounds_s,
        population_size=args.population,
        generations=args.generations,
        seeds=args.seeds,
        evaluation_duration_s=args.evaluation_duration,
        speed_factor=args.speed_factor,
        progress_callback=log_generation,
        random_seed=args.random_seed,
    )
    print(
        "Training six-phase neural policy "
        f"({args.population} candidates, {args.generations} generations, "
        f"seeds={args.seeds}, speed={args.speed_factor:g}x)..."
    )
    result = trainer.run()
    best = result["best"]
    output = {
        "format_version": 1,
        "policy_type": "six_phase",
        "phases": list(PHASE_NAMES),
        "network": {
            "input_size": SixPhasePolicy.input_size,
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
        "history": result["history"],
        "training": {
            "population_size": args.population,
            "generations": args.generations,
            "seeds": args.seeds,
            "evaluation_duration_s": args.evaluation_duration,
            "speed_factor": args.speed_factor,
            "training_time_s": result["training_time_s"],
            "fitness_weights": dict(CONFIG.get("fitness", {})),
            "six_phase_fitness_weights": dict(
                CONFIG.get("six_phase_fitness", {})
            ),
            "random_seed": args.random_seed,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Best fitness: {best['fitness']:.2f}")
    print(f"Total training time: {result['training_time_s']:.2f}s")
    print(f"Saved six-phase policy to: {args.output}")


if __name__ == "__main__":
    main()
