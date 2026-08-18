"""Calibrate fixed green durations without using holdout scenarios."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
from statistics import fmean, pstdev

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from config import CONFIG, build_runtime_config
from simulation.evaluation import evaluate_fixed_time_policy_across_seeds
from simulation.fixed_time import FixedTimeMovementPlan, load_fixed_time_plan


def parse_seeds(value):
    try:
        seeds = tuple(
            int(item.strip()) for item in value.split(",") if item.strip()
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "seeds must be comma-separated integers"
        ) from error
    if not seeds:
        raise argparse.ArgumentTypeError("provide at least one seed")
    return seeds


def _positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _positive_float(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive and finite")
    return value


def _nonnegative_float(value):
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return value


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate only the stage durations of a version-1 fixed-time "
            "movement plan on configured training scenarios."
        )
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--population", type=_positive_int, default=13)
    parser.add_argument("--generations", type=_positive_int, default=12)
    parser.add_argument("--min-green", type=_positive_float, default=5.0)
    parser.add_argument("--max-green", type=_positive_float, default=45.0)
    parser.add_argument(
        "--mutation-sigma",
        type=_positive_float,
        default=3.0,
        help="initial standard deviation of duration mutations in seconds",
    )
    parser.add_argument("--seeds", type=parse_seeds, default=(1, 2, 3))
    parser.add_argument(
        "--evaluation-duration",
        type=_positive_float,
        default=90.0,
    )
    parser.add_argument("--timestep", type=_positive_float, default=1 / 30)
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=max(1, min(8, (os.cpu_count() or 1) - 1)),
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--robustness-penalty",
        type=_nonnegative_float,
        default=0.25,
        help="robust score = scenario mean - penalty * scenario stddev",
    )
    args = parser.parse_args(argv)
    if args.min_green > args.max_green:
        parser.error("--min-green cannot exceed --max-green")
    return args


def clone_plan_with_durations(plan, durations_s):
    """Return a validated clone while preserving names and movement sets."""
    durations_s = tuple(float(value) for value in durations_s)
    if len(durations_s) != len(plan.stages):
        raise ValueError("one duration is required for every fixed-time stage")
    payload = plan.to_dict()
    for stage_payload, duration_s in zip(payload["stages"], durations_s):
        stage_payload["duration_s"] = duration_s
    return FixedTimeMovementPlan.from_dict(payload)


def score_evaluation(evaluation, robustness_penalty):
    """Summarize common-scenario fitness with a robustness adjustment."""
    samples = [
        float(item["fitness"])
        for item in evaluation.get("evaluations", ())
    ]
    skipped_count = int(evaluation.get("skipped_scenario_count", 0))
    if skipped_count:
        imputed = samples[-1] if samples else float(evaluation["mean_fitness"])
        samples.extend(imputed for _ in range(skipped_count))
    if not samples:
        samples = [float(evaluation["mean_fitness"])]
    if any(not math.isfinite(value) for value in samples):
        raise ValueError("evaluation returned a non-finite fitness")
    mean_fitness = fmean(samples)
    fitness_std = pstdev(samples) if len(samples) > 1 else 0.0
    return {
        "scenario_mean_fitness": mean_fitness,
        "scenario_fitness_std": fitness_std,
        "robust_fitness": (
            mean_fitness - float(robustness_penalty) * fitness_std
        ),
        "scenario_count": len(samples),
    }


def _evaluate_candidate(payload):
    candidate = clone_plan_with_durations(
        payload["plan"],
        payload["durations_s"],
    )
    evaluation = evaluate_fixed_time_policy_across_seeds(
        payload["config"],
        candidate,
        seeds=payload["seeds"],
        duration_s=payload["evaluation_duration_s"],
        timestep_s=payload["timestep_s"],
        traffic_profiles=payload["traffic_profiles"],
        abort_remaining_seeds_on_gridlock=False,
    )
    return {
        "durations_s": tuple(payload["durations_s"]),
        **score_evaluation(evaluation, payload["robustness_penalty"]),
    }


def _bounded(value, lower, upper):
    return min(upper, max(lower, value))


def _generation_candidates(
    incumbent,
    population_size,
    sigma_s,
    minimum_green_s,
    maximum_green_s,
    rng,
):
    """Mix deterministic coordinate probes with Gaussian population search."""
    incumbent = tuple(incumbent)
    candidates = [incumbent]
    for stage_index in range(len(incumbent)):
        for direction in (1.0, -1.0):
            if len(candidates) >= population_size:
                return candidates
            candidate = list(incumbent)
            candidate[stage_index] = _bounded(
                candidate[stage_index] + direction * sigma_s,
                minimum_green_s,
                maximum_green_s,
            )
            candidates.append(tuple(candidate))
    while len(candidates) < population_size:
        candidates.append(
            tuple(
                _bounded(
                    duration_s + rng.gauss(0.0, sigma_s),
                    minimum_green_s,
                    maximum_green_s,
                )
                for duration_s in incumbent
            )
        )
    return candidates


def calibrate_plan(
    config,
    plan,
    traffic_profiles,
    seeds=(1, 2, 3),
    population_size=13,
    generations=12,
    minimum_green_s=5.0,
    maximum_green_s=45.0,
    mutation_sigma_s=3.0,
    evaluation_duration_s=90.0,
    timestep_s=1 / 30,
    workers=1,
    random_seed=42,
    robustness_penalty=0.25,
    progress_callback=None,
):
    """Calibrate stage durations on training scenarios only."""
    if minimum_green_s > maximum_green_s:
        raise ValueError("minimum_green_s cannot exceed maximum_green_s")
    if population_size <= 0 or generations <= 0 or workers <= 0:
        raise ValueError("population_size, generations, and workers must be positive")
    seeds = tuple(seeds)
    traffic_profiles = tuple(traffic_profiles) or ({"name": "configured"},)
    if not seeds:
        raise ValueError("at least one training seed is required")

    rng = random.Random(random_seed)
    incumbent = tuple(
        _bounded(
            stage.duration_s,
            minimum_green_s,
            maximum_green_s,
        )
        for stage in plan.stages
    )

    def make_payload(durations_s):
        return {
            "config": config,
            "plan": plan,
            "durations_s": durations_s,
            "seeds": seeds,
            "traffic_profiles": traffic_profiles,
            "evaluation_duration_s": evaluation_duration_s,
            "timestep_s": timestep_s,
            "robustness_penalty": robustness_penalty,
        }

    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        best = _evaluate_candidate(make_payload(incumbent))
        for generation in range(1, generations + 1):
            sigma_s = mutation_sigma_s * (0.85 ** (generation - 1))
            durations_population = _generation_candidates(
                incumbent,
                population_size,
                sigma_s,
                minimum_green_s,
                maximum_green_s,
                rng,
            )
            payloads = [make_payload(durations) for durations in durations_population]
            if executor is None:
                evaluated = [_evaluate_candidate(payload) for payload in payloads]
            else:
                evaluated = list(executor.map(_evaluate_candidate, payloads))
            generation_best = max(
                evaluated,
                key=lambda item: (
                    item["robust_fitness"],
                    item["scenario_mean_fitness"],
                ),
            )
            if generation_best["robust_fitness"] > best["robust_fitness"]:
                best = generation_best
                incumbent = best["durations_s"]
            if progress_callback is not None:
                progress_callback(generation, generations, best)
    finally:
        if executor is not None:
            executor.shutdown()

    return {
        "plan": clone_plan_with_durations(plan, best["durations_s"]),
        **best,
    }


def build_calibrated_output_plan(plan, calibration_result, config, args, profiles):
    """Attach reproducibility metadata without running any holdout evaluation."""
    payload = calibration_result["plan"].to_dict()
    metadata = deepcopy(payload.get("metadata", {}))
    metadata["calibration"] = {
        "method": "bounded_coordinate_gaussian_search",
        "training_seeds": list(args.seeds),
        "training_profiles": deepcopy(list(profiles)),
        "evaluation_duration_s": args.evaluation_duration,
        "timestep_s": args.timestep,
        "population": args.population,
        "generations": args.generations,
        "minimum_green_s": args.min_green,
        "maximum_green_s": args.max_green,
        "mutation_sigma_s": args.mutation_sigma,
        "workers": args.workers,
        "random_seed": args.random_seed,
        "robustness_penalty": args.robustness_penalty,
        "scenario_mean_fitness": calibration_result["scenario_mean_fitness"],
        "scenario_fitness_std": calibration_result["scenario_fitness_std"],
        "robust_fitness": calibration_result["robust_fitness"],
        "scenario_count": calibration_result["scenario_count"],
        "fitness": {
            "base": deepcopy(config.get("fitness", {})),
            "movement": deepcopy(config.get("six_phase_fitness", {})),
        },
        "holdout_evaluated": False,
        "holdout_untouched": True,
    }
    payload["metadata"] = metadata
    return FixedTimeMovementPlan.from_dict(payload)


def main(argv=None):
    args = parse_arguments(argv)
    runtime_config = build_runtime_config(CONFIG)
    plan = load_fixed_time_plan(args.plan)
    profiles = tuple(
        runtime_config.get("six_phase_training", {}).get(
            "traffic_profiles",
            (),
        )
    ) or ({"name": "configured"},)

    print(
        "Calibrating fixed-time plan "
        f"({args.population} candidates, {args.generations} generations, "
        f"{len(profiles)} profiles, seeds={args.seeds}, workers={args.workers})...",
        flush=True,
    )

    def log_progress(generation, total_generations, best):
        print(
            f"Generation {generation}/{total_generations} | "
            f"mean={best['scenario_mean_fitness']:.2f} | "
            f"std={best['scenario_fitness_std']:.2f} | "
            f"robust={best['robust_fitness']:.2f}",
            flush=True,
        )

    result = calibrate_plan(
        runtime_config,
        plan,
        profiles,
        seeds=args.seeds,
        population_size=args.population,
        generations=args.generations,
        minimum_green_s=args.min_green,
        maximum_green_s=args.max_green,
        mutation_sigma_s=args.mutation_sigma,
        evaluation_duration_s=args.evaluation_duration,
        timestep_s=args.timestep,
        workers=args.workers,
        random_seed=args.random_seed,
        robustness_penalty=args.robustness_penalty,
        progress_callback=log_progress,
    )
    calibrated_plan = build_calibrated_output_plan(
        plan,
        result,
        runtime_config,
        args,
        profiles,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(calibrated_plan.to_dict(), stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        f"Saved calibrated fixed-time plan to {args.output} | "
        f"robust fitness={result['robust_fitness']:.2f}",
        flush=True,
    )
    return result


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
