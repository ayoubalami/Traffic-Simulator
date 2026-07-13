"""Renderer-free evaluation utilities for traffic-signal optimization."""

from copy import deepcopy
from statistics import fmean

from .simulation import Simulation


def calculate_fitness(metrics):
    """Return a score that rewards exits and penalizes delay and queues."""
    queue_size = sum(metrics["queue_lengths"].values())
    return (
        metrics["throughput"] * 100.0
        - metrics["avg_wait_time"] * 10.0
        - metrics["avg_active_wait_time"] * 5.0
        - metrics["max_wait_time"]
        - queue_size * 10.0
    )


def evaluate_signal_timings(
    config,
    green_durations_s,
    duration_s=300.0,
    timestep_s=1 / 30,
    random_seed=1,
):
    """Evaluate one four-direction timing candidate without a renderer.

    ``green_durations_s`` is a mapping such as ``{"north": 12.0, ...}``.
    Use the same seed and scenario when comparing candidate solutions.
    """
    if duration_s <= 0 or timestep_s <= 0:
        raise ValueError("duration_s and timestep_s must be positive")

    evaluation_config = deepcopy(config)
    timing = evaluation_config.setdefault("traffic_lights", {})
    durations = timing.setdefault("green_durations_s", {})
    durations.update(green_durations_s)

    simulation = Simulation(evaluation_config, random_seed=random_seed)
    elapsed = 0.0
    while elapsed < duration_s:
        dt = min(timestep_s, duration_s - elapsed)
        simulation.update(dt)
        elapsed += dt

    metrics = simulation.metrics.get_summary()
    return {
        "fitness": calculate_fitness(metrics),
        "metrics": metrics,
        "green_durations_s": dict(durations),
        "random_seed": random_seed,
    }


def evaluate_neural_policy(
    config,
    policy,
    duration_s=300.0,
    timestep_s=1 / 30,
    random_seed=1,
):
    """Evaluate a queue-driven duration policy without rendering."""
    if duration_s <= 0 or timestep_s <= 0:
        raise ValueError("duration_s and timestep_s must be positive")

    simulation = Simulation(
        deepcopy(config),
        random_seed=random_seed,
        extension_decider=policy.should_extend,
    )
    elapsed = 0.0
    while elapsed < duration_s:
        dt = min(timestep_s, duration_s - elapsed)
        simulation.update(dt)
        elapsed += dt

    metrics = simulation.metrics.get_summary()
    return {
        "fitness": calculate_fitness(metrics),
        "metrics": metrics,
        "random_seed": random_seed,
    }


def evaluate_across_seeds(
    config,
    green_durations_s,
    seeds=(1, 2, 3),
    duration_s=300.0,
    timestep_s=1 / 30,
):
    """Score one timing plan across fixed scenarios and average the results."""
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError("at least one random seed is required")

    evaluations = [
        evaluate_signal_timings(
            config,
            green_durations_s,
            duration_s=duration_s,
            timestep_s=timestep_s,
            random_seed=seed,
        )
        for seed in seeds
    ]
    metric_names = (
        "throughput",
        "avg_wait_time",
        "avg_active_wait_time",
        "avg_travel_time",
        "max_wait_time",
        "active_vehicles",
    )

    return {
        "mean_fitness": fmean(result["fitness"] for result in evaluations),
        "mean_metrics": {
            name: fmean(result["metrics"][name] for result in evaluations)
            for name in metric_names
        },
        "green_durations_s": dict(green_durations_s),
        "seeds": seeds,
        "evaluations": evaluations,
    }


def evaluate_policy_across_seeds(
    config,
    policy,
    seeds=(1, 2, 3),
    duration_s=300.0,
    timestep_s=1 / 30,
):
    """Score a neural duration policy across fixed random seeds."""
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError("at least one random seed is required")

    evaluations = [
        evaluate_neural_policy(
            config,
            policy,
            duration_s=duration_s,
            timestep_s=timestep_s,
            random_seed=seed,
        )
        for seed in seeds
    ]
    metric_names = (
        "throughput",
        "avg_wait_time",
        "avg_active_wait_time",
        "avg_travel_time",
        "max_wait_time",
        "active_vehicles",
    )
    return {
        "mean_fitness": fmean(result["fitness"] for result in evaluations),
        "mean_metrics": {
            name: fmean(result["metrics"][name] for result in evaluations)
            for name in metric_names
        },
        "seeds": seeds,
        "evaluations": evaluations,
    }
